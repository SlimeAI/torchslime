from torchslime.logging.logger import logger
from .common import list_take
from .typing import (
    Sequence,
    Tuple,
    Any,
    Union,
    TYPE_CHECKING
)
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from torchslime.core.context import Context


class DataProvider:

    def __init__(self):
        pass

    def get(self, ctx: "Context") -> DataLoader:
        pass

    def __call__(self, ctx: "Context") -> DataLoader:    
        data_loader = self.get(ctx)
        if isinstance(data_loader, DataLoader) is False:
            logger.warning('DataProvider returns a non-DataLoader object, this may cause some problems.')
        return data_loader


class ConstantProvider(DataProvider):

    def __init__(self, dataset: DataLoader):
        super().__init__()
        self.dataset = dataset

    def get(self, _) -> DataLoader:
        return self.dataset


class DataParser:

    def __init__(self):
        pass

    def get(self, ctx: "Context") -> Tuple[Any, Any, Any]: pass

    def __call__(self, ctx: "Context") -> Tuple[Any, Any, Any]:
        batch = self.get(ctx)
        if isinstance(batch, tuple) is False or len(batch) != 3:
            logger.warning('DataParser returns a non-tuple object or the tuple length is not 3, this may cause value-unpack exceptions.')
        return batch


class IndexParser(DataParser):

    def __init__(
        self,
        x: Union[Sequence[int], int] = 0,
        y: Union[Sequence[int], int] = 1,
        extra: Union[Sequence[int], int] = None
    ):
        super(IndexParser, self).__init__()
        self.x = x
        self.y = y
        self.extra = extra

    def get(self, ctx: "Context") -> Tuple[Any, Any, Any]:
        batch = ctx.step_ctx.batch
        return list_take(batch, self.x), list_take(batch, self.y), list_take(batch, self.extra)
