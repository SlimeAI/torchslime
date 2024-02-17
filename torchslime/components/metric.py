from torchslime.utils.common import (
    LessThanAnything,
    dict_merge,
    safe_divide,
    GreaterThanAnything
)
from torchslime.utils.typing import (
    NOTHING,
    Mapping,
    NoneOrNothing,
    Union,
    Dict,
    Iterable,
    Callable,
    Any,
    is_none_or_nothing,
    TYPE_CHECKING
)
from torchslime.utils.bases import BaseDict, BaseList
from .registry import Registry
from .exception import APIMisused
from torchslime.logging.logger import logger
from torch import Tensor
import inspect
if TYPE_CHECKING:
    from torchslime.core.context import Context


class Metric:

    def __init__(self, name: Union[str, NoneOrNothing] = None):
        self.name = name

    def get(self, ctx: "Context") -> Union[Dict, int, float]: pass

    def __call__(self, ctx: "Context") -> Dict:
        result = self.get(ctx)
        # metric dict
        if isinstance(result, Dict):
            return result
        # metric value
        if is_none_or_nothing(self.name):
            classname = str(self.__class__.__name__)
            raise APIMisused(
                f'When ``{classname}`` returns non-dict value, '
                'param ``name`` should be specified and cannot be ``None`` or ``NOTHING``.'
            )
        return { self.name: result }


class MetricContainer(Metric, BaseList[Metric]):

    def __init__(self, metrics: Iterable[Metric] = None):
        Metric.__init__(self)
        BaseList.__init__(self, metrics)

    def get(self, ctx: "Context") -> Dict:
        result = {}
        for metric in self:
            _res = metric(ctx)
            result = dict_merge(result, _res)
        return result


class LossFunc(Metric):
    
    def get(self, ctx: "Context") -> Union[Dict, Tensor]: pass


class SimpleLossFunc(LossFunc):
    
    def __init__(
        self,
        loss_func: Callable[[Any, Any], Union[Dict, Tensor]],
        name: Union[str, NoneOrNothing] = None
    ):
        super().__init__(name)
        self.loss_func = loss_func
    
    def get(self, ctx: "Context") -> Union[Dict, Tensor]:
        return self.loss_func(ctx.step_ctx.y_pred, ctx.step_ctx.y_true)


class LossFuncContainer(MetricContainer):
    
    def __init__(self, loss_func_list: Iterable[LossFunc] = None):
        if not is_none_or_nothing(loss_func_list) and not isinstance(loss_func_list, Iterable):
            raise ValueError('Param ``loss_func_list`` should be a list.')
        super().__init__(loss_func_list)
    
    def __call__(self, ctx: "Context") -> Dict:
        # if there is only one LossFunc, set the name to 'loss'
        if len(self) == 1 and isinstance(self[0], Metric) and is_none_or_nothing(self[0].name):
            self[0].name = 'loss'
        return super().__call__(ctx)


class LossReductionFactory:

    @staticmethod
    def get(
        item: Union[str, dict, Callable[["Context"], Tensor]]
    ) -> Callable[["Context"], Tensor]:
        if isinstance(item, str):
            if not item in loss_reduction_registry:
                raise ValueError('Loss reduction type not supported.')
            return loss_reduction_registry.get(item)
        elif isinstance(item, dict):
            # weighted loss reduction
            return _weighted_loss_reduction(item)
        elif inspect.isfunction(item) or inspect.ismethod(item):
            # user defined function
            return item
        else:
            raise NotImplementedError('Loss reduction type not supported. '
                'Use str, dict or (Context) -> Tensor function instead.')


loss_reduction_registry = Registry('loss_reduction_registry')


@loss_reduction_registry(name='mean')
def _mean_loss_reduction(ctx: "Context"):
    loss_tensors = ctx.step_ctx.loss.values()
    tensor_len = len(loss_tensors)
    result = sum(map(lambda loss_tensor: safe_divide(loss_tensor, tensor_len, NOTHING), loss_tensors))
    if result is NOTHING:
        logger.warning(
            'Mean loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
            '1. Values returned by the loss func contain NOTHING.\n'
            '2. Length of loss values is 0.'
        )
    return result


@loss_reduction_registry(name='sum')
def _sum_loss_reduction(ctx: "Context"):
    loss_tensors = ctx.step_ctx.loss.values()
    result = sum(loss_tensors) if len(loss_tensors) > 0 else NOTHING
    if result is NOTHING:
        logger.warning(
            'Sum loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
            '1. Values returned by the loss func contain NOTHING.\n'
            '2. Length of loss values is 0.'
        )
    return result


def _weighted_loss_reduction(weight: dict):
    _weight = dict(weight)
    def _reduction(ctx: "Context"):
        loss_dict = ctx.step_ctx.loss
        # check keys intersection
        loss_keys = set(loss_dict.keys())
        weight_keys = set(_weight.keys())
        common_keys = loss_keys & weight_keys
        # TODO: unused loss warning
        result = 0 if len(common_keys) > 0 else NOTHING
        for key in list(common_keys):
            if key in loss_dict:
                result += _weight[key] * loss_dict[key]
        if result is NOTHING:
            logger.warning(
                'Weighted loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
                '1. Values returned by the loss func contain NOTHING.\n'
                '2. Weight values contain NOTHING.\n'
                '3. There are no matched keys between loss keys and weight keys.'
            )
        return result
    return _reduction


class Meter:
    
    def __init__(self) -> None:
        self.initialize()

    def initialize(self) -> None:
        self.count = 0
        self.min = GreaterThanAnything()
        self.max = LessThanAnything()
        self.mean = 0
    
    def __call__(self, __value) -> None:
        # use ``self.min > xxx`` here (rather than ``xxx < self.min``) to make sure to apply ``__gt__`` in ``GreaterThanAnything``
        if self.min > __value:
            self.min = __value
        if self.max < __value:
            self.max = __value
        self.mean = self.mean * (self.count / (self.count + 1)) + __value / (self.count + 1)
        self.count += 1
    
    def get__(self, __key: Union[str, NoneOrNothing] = None):
        result = {
            'mean': self.mean,
            'min': self.min,
            'max': self.max,
            'count': self.count
        }
        return result if is_none_or_nothing(__key) else result.get(__key, NOTHING)


class MeterDict(BaseDict[str, Meter]):
    
    def __call__(self, __value: Mapping[str, Any]) -> None:
        for key, value in __value.items():
            if key not in self:
                self[key] = Meter()
            # compute meter
            self[key](value)

    def get__(self, __key: Union[str, NoneOrNothing] = None) -> Dict:
        result = {}
        for key, meter in self.items():
            result[key] = meter.get__(__key)
        return result
