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
    Type
)
from torchslime.utils.common import dict_to_key_value_str
from torchslime.utils.registry import Registry
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
    
    def parse_meter_dict(
        self,
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


@profiler_registry(name='vanilla')
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
        if is_none_or_nothing(state):
            state = ctx.pipeline_ctx.model_state
        elif isinstance(state, str):
            state = state_registry.get(state)()
        
        loss_values, metrics = state.get_meter(ctx)
        loss_value_str = self.parse_meter_dict(loss_values.get__('mean'))
        metric_str = self.parse_meter_dict(metrics.get__('mean'))
        
        return f'[{str(state)} Profile] | [Loss] {loss_value_str} | [Metrics] {metric_str}'
