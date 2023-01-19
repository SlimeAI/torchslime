from abc import abstractmethod
from typing import Union, Dict, Sequence
from torchslime.util.type import NUMBER, NUMBER_T
from torchslime.util import Count, Nothing, is_nothing, dict_merge, NOTHING, BaseList, BaseDict
from torchslime.core.context import Context


class Metric:

    count = Count()
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def get(self, ctx: Context) -> Union[Dict, NUMBER]:
        pass

    def __call__(self, ctx: Context) -> Union[Dict, Nothing]:
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

    def get(self, ctx: Context) -> Union[Dict, NUMBER]:
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

        pass

    def get(self, loss):
        pass

