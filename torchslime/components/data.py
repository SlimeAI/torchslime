from torchslime.core.context.base import BaseContext
from torchslime.utils import list_take
from torchslime.logging.logger import logger
from torchslime.utils.typing import (
    Sequence,
    Tuple,
    Any,
    Union,
    TYPE_CHECKING
)
# Type check only
if TYPE_CHECKING:
    from torchslime.utils.typing import (
        TorchDataLoader
    )


class DataProvider:

    def __init__(self):
        pass

    def get(self, ctx: BaseContext) -> "TorchDataLoader":
        pass

    def __call__(self, ctx: BaseContext) -> "TorchDataLoader":
        from torch.utils.data import DataLoader
        
        data_loader = self.get(ctx)
        if isinstance(data_loader, DataLoader) is False:
            logger.warning('DataProvider returns a non-DataLoader object, this may cause some problems.')
        return data_loader


class ConstantProvider(DataProvider):

    def __init__(self, dataset: "TorchDataLoader"):
        super().__init__()
        self.dataset = dataset

    def get(self, _) -> "TorchDataLoader":
        return self.dataset


class DataParser:

    def __init__(self):
        pass

    def get(self, ctx: BaseContext) -> Tuple[Any, Any, Any]: pass

    def __call__(self, ctx: BaseContext) -> Tuple[Any, Any, Any]:
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

    def get(self, ctx: BaseContext) -> Tuple[Any, Any, Any]:
        batch = ctx.step_ctx.batch
        return list_take(batch, self.x), list_take(batch, self.y), list_take(batch, self.extra)
