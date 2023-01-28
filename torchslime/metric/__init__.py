from abc import abstractmethod
from typing import Union, Dict, Sequence, Callable
from torchslime.util.type import NUMBER, NUMBER_T
from torchslime.util import Count, Nothing, is_nothing, dict_merge, NOTHING, BaseList, BaseDict, safe_divide
from torchslime.core.context import BaseContext
from torchslime.log import logger
from torch import Tensor
import inspect


class Metric:

    count = Count()
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def get(self, ctx: BaseContext) -> Union[Dict, NUMBER]:
        pass

    def __call__(self, ctx: BaseContext) -> Union[Dict, Nothing]:
        result = self.get(ctx)
        if isinstance(result, Dict):
            return result
        elif isinstance(result, NUMBER_T):
            if self.name is None:
                # use default name
                self.name = 'metric_%d' % self.count
            return { self.name: result }
        return NOTHING


# metric callback or sequence of metric callbacks
M_SEQ = Union[Metric, Sequence[Metric]]


class MetricContainer(Metric, BaseList):

    def __init__(self, metrics: M_SEQ = None):
        Metric.__init__(self)
        BaseList.__init__(self, metrics)

    def get(self, ctx: BaseContext) -> Union[Dict, NUMBER]:
        result = {}
        for metric in self:
            _res = metric(ctx)
            # is not Nothing
            if is_nothing(_res) is False:
                # TODO: change the dict merge operation
                result = dict_merge(result, _res)
        return result


class LossParser:

    class LossWrapper(BaseDict):

        def __init__(self, loss_dict: Dict, wrapped: bool):
            super().__init__(loss_dict)
            self.__wrapped = wrapped

        def decode(self):
            return self._BaseDict__dict['loss'] if self.__wrapped is True else self._BaseDict__dict

    def get(self, loss) -> LossWrapper:
        is_dict_loss = self.is_dict_loss(loss)
        return self.LossWrapper(
            loss if is_dict_loss is True else {'loss': loss},
            not is_dict_loss
        )

    def get_copy(self, loss) -> LossWrapper:
        return self.get(dict(loss) if self.is_dict_loss(loss) is True else loss)

    def is_dict_loss(self, loss):
        return isinstance(loss, (dict, Dict))


class LossReductionFactory:

    @staticmethod
    def get(
        item: Union[str, dict, Callable[[BaseContext], Tensor]]
    ) -> Callable[[BaseContext], Tensor]:
        if isinstance(item, str):
            str_mapper = {
                'mean': _mean_loss_reduction,
                'sum': _sum_loss_reduction
            }
            assert item in list(str_mapper.keys()), 'Loss reduction type not supported.'
            return str_mapper[item]
        elif isinstance(item, (dict, Dict)):
            # weighted loss reduction
            return _weighted_loss_reduction(item)
        elif inspect.isfunction(item) or inspect.ismethod(item):
            # user defined function
            return item
        else:
            raise NotImplementedError('Loss reduction type not supported. '
                'Use str, dict or (Context) -> Tensor function instead.')


def _mean_loss_reduction(ctx: BaseContext):
    loss_tensors = ctx.run.loss_parser.get(ctx.step.loss).values()
    result = safe_divide(sum(loss_tensors), len(loss_tensors), NOTHING)
    if is_nothing(result):
        logger.warn('Mean loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
            '1. Values returned by the loss func contain NOTHING.\n'
            '2. Length of loss values is 0.')
    return result


def _sum_loss_reduction(ctx: BaseContext):
    loss_tensors = ctx.run.loss_parser.get(ctx.step.loss).values()
    result = sum(loss_tensors) if len(loss_tensors) > 0 else NOTHING
    if is_nothing(result):
        logger.warn('Sum loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
            '1. Values returned by the loss func contain NOTHING.\n'
            '2. Length of loss values is 0.')
    return result


def _weighted_loss_reduction(weight: dict):
    _weight = dict(weight)
    def _reduction(ctx: BaseContext):
        loss_dict = ctx.run.loss_parser.get(ctx.step.loss)
        # check keys intersection
        loss_keys = set(loss_dict.keys())
        weight_keys = set(_weight.keys())
        common_keys = loss_keys & weight_keys
        # TODO: unused loss warning
        result = 0 if len(common_keys) > 0 else NOTHING
        for key in list(common_keys):
            if key in loss_dict:
                result += _weight[key] * loss_dict[key]
        if is_nothing(result):
            logger.warn('Weighted loss reduction got NOTHING. This may be caused by one of the following reasons:\n'
                '1. Values returned by the loss func contain NOTHING.\n'
                '2. Weight values contain NOTHING.\n'
                '3. There are no matched keys between loss keys and weight keys.')
        return result
    return _reduction
