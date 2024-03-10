"""
Analyzing context profile.
"""
from torchslime.utils.typing import (
    TYPE_CHECKING,
    Union,
    Any,
    Dict,
    NoneOrNothing,
    is_none_or_nothing,
    NOTHING,
    Type,
    Callable,
    MISSING,
    Missing,
    Tuple
)
from torchslime.utils.common import dict_to_key_value_str
from torchslime.utils.registry import Registry
from torchslime.utils.exception import APIMisused
from functools import partial
from .state import state_registry

if TYPE_CHECKING:
    from torchslime.context import Context
    from .state import ModelState

profiler_registry = Registry[Type["PipelineProfiler"]]('profiler_registry')


class PipelineProfiler:
    
    def logging_point_profile(self, ctx: "Context") -> str: pass
    def meter_profile(
        self,
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> str: pass
    def logging_profile(
        self,
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> str: pass
    
    def get_logging_profile_func_dict(
        self,
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> Dict[str, Callable[[], str]]:
        """
        Returns profile function dict that may be called in ``logging_profile``. Use 
        this method to improve running efficiency, because only specified functions are 
        called according to profile values needed to be computed.
        """
        return {
            'logging_point': partial(self.logging_point_profile, ctx),
            'meter': partial(self.meter_profile, ctx, state)
        }
    
    @staticmethod
    def parse_meter_dict(
        meter: Dict[str, Any],
        format: str = '.5f',
        key_value_sep: str = ': ',
        str_sep: str = ' - ',
        placeholder: Union[str, NoneOrNothing] = '(Empty)'
    ) -> str:
        if len(meter) < 1 and not is_none_or_nothing(placeholder):
            # set placeholder if profile is empty
            profile_str = placeholder
        else:
            profile_str = dict_to_key_value_str(
                {k:f'{v:{format}}' for k, v in meter.items()},
                key_value_sep=key_value_sep,
                str_sep=str_sep
            )
        return profile_str

    @staticmethod
    def parse_state(
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> "ModelState":
        if is_none_or_nothing(state):
            state = ctx.pipeline_ctx.model_state
        elif isinstance(state, str):
            state = state_registry.get(state)()
        return state


@profiler_registry(key='vanilla')
class VanillaProfiler(PipelineProfiler):
    
    def logging_point_profile(self, ctx: "Context") -> str:
        build_name = str(ctx.hook_ctx.build)
        current = ctx.iteration_ctx.current + 1
        total = ctx.iteration_ctx.total
        return f'[{build_name} {current}/{total}]'

    def meter_profile(
        self,
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> str:
        state = self.parse_state(ctx, state)
        
        loss_values, metrics = state.get_meter(ctx)
        loss_value_str = self.parse_meter_dict(loss_values.get__('mean'))
        metric_str = self.parse_meter_dict(metrics.get__('mean'))
        
        return f'[{str(state)} Profile] | [Loss] {loss_value_str} | [Metrics] {metric_str}'
    
    def logging_profile(
        self,
        ctx: "Context",
        state: Union[str, "ModelState", NoneOrNothing] = NOTHING
    ) -> str:
        """
        Get logging profile.
        """
        state = self.parse_state(ctx, state)
        # Get profile keys.
        profile_keys: Union[Tuple[str, ...], Missing] = getattr(
            state.format_logging_profile, 'profile_keys__', MISSING
        )
        # Partial functions that may be called.
        profile_func_dict = self.get_logging_profile_func_dict(ctx, state)
        
        profile_dict = {}
        if profile_keys is MISSING:
            # If ``profile_keys__`` is not specified, call all the functions.
            for key, value in profile_func_dict.items():
                profile_dict[key] = value
        else:
            # Else call specified functions.
            for key in profile_keys:
                if key not in profile_func_dict:
                    raise APIMisused(
                        f'The given profile key is not contained in the ``profile_func_dict``. '
                        f'Given key: {key}, Acceptable keys: {tuple(profile_func_dict.keys())}'
                    )
                profile_dict[key] = profile_func_dict[key]()
        return state.format_logging_profile(ctx, profile_dict)
