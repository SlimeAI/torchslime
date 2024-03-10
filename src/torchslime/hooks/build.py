from torchslime.utils.registry import Registry
from torchslime.handlers.wrapper import validation_check
from torchslime.logging.logger import logger
from torchslime.utils.typing import (
    Generator,
    TYPE_CHECKING
)
from torchslime.utils.base import (
    BaseGenerator,
    BaseGeneratorQueue
)
from slime_core.hooks.build import (
    CoreBuildHook,
    CoreBuildInterface
)
if TYPE_CHECKING:
    from torchslime.context import Context

build_registry = Registry('build_registry')


class BuildHook(CoreBuildHook["Context"]):

    def build_train(self, ctx: "Context") -> None: pass
    def build_eval(self, ctx: "Context") -> None: pass
    def build_predict(self, ctx: "Context") -> None: pass

    def run_build_train__(self, ctx: "Context"):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx
        
        with BaseGeneratorQueue((
            BaseGenerator(h.launch.build_train_yield(ctx)),
            BaseGenerator(h.plugins.build_train_yield(ctx))
        )):
            h.build.build_train(ctx)
    
    def run_build_eval__(self, ctx: "Context"):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx
        
        with BaseGeneratorQueue((
            BaseGenerator(h.launch.build_eval_yield(ctx)),
            BaseGenerator(h.plugins.build_eval_yield(ctx))
        )):
            h.build.build_eval(ctx)

    def run_build_predict__(self, ctx: "Context"):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx

        with BaseGeneratorQueue((
            BaseGenerator(h.launch.build_predict_yield(ctx)),
            BaseGenerator(h.plugins.build_predict_yield(ctx))
        )):
            h.build.build_predict(ctx)


class BuildInterface(CoreBuildInterface["Context"]):
    """
    Interface for building handlers.
    """
    def build_train_yield(self, ctx: "Context") -> Generator: yield
    def build_eval_yield(self, ctx: "Context") -> Generator: yield
    def build_predict_yield(self, ctx: "Context") -> Generator: yield


@build_registry(key='vanilla')
class VanillaBuild(BuildHook):
    
    def build_train(self, ctx: "Context"):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.pipeline_ctx.train_container = handler.RootContainer(
            id='container',
            handlers=[
                # epoch iter
                handler.EpochIterationContainer(
                    id='epoch_iteration',
                    handlers=[
                        # train part
                        handler.Container(
                            id='container_train',
                            wrappers=[
                                # set state to train
                                handler.StateWrapper(id='state_train', state='train')
                            ],
                            handlers=[
                                # init meter setting
                                handler.MeterInitHandler(id='meter_init_train'),
                                # dataset iter
                                handler.IterationContainer(
                                    id='iteration_train',
                                    handlers=[
                                        # forward
                                        handler.ForwardHandler(id='forward_train'),
                                        # compute loss
                                        handler.LossHandler(id='loss_train'),
                                        # backward and optimizer step
                                        handler.OptimizerContainer(
                                            id='optimizer_train',
                                            handlers=[
                                                handler.BackwardHandler(id='backward_train')
                                            ]
                                        ),
                                        # compute metrics
                                        handler.MetricHandler(id='metrics_train'),
                                        # compute meter loss value and metrics
                                        handler.MeterHandler(id='meter_train'),
                                        # apply learning rate schedule
                                        handler.LRScheduleHandler(id='lr_schedule')
                                    ]
                                ),
                                # logging
                                handler.LoggingHandler(['train'], id='logging_train')
                            ]
                        ),
                        # validation part
                        handler.Container(
                            id='container_val',
                            wrappers=[
                                # validation according to valid_seq
                                handler.ConditionWrapper(
                                    id='condition_val',
                                    condition=validation_check
                                ),
                                # set state to val
                                handler.StateWrapper(id='state_val', state='val')
                            ],
                            handlers=[
                                handler.FuncHandler(
                                    id='logging_val_start',
                                    exec_ranks=[0],
                                    func_list=[
                                        lambda _: logger.info('Validation starts.')
                                    ]
                                ),
                                # init meter setting
                                handler.MeterInitHandler(id='meter_init_val'),
                                # dataset iter
                                handler.IterationContainer(
                                    id='iteration_val',
                                    handlers=[
                                        # forward
                                        handler.ForwardHandler(id='forward_val'),
                                        # compute loss
                                        handler.LossHandler(id='loss_val'),
                                        # metrics
                                        handler.MetricHandler(id='metrics_val'),
                                        # compute meter loss value and metrics
                                        handler.MeterHandler(id='meter_val')
                                    ]
                                ),
                                # logging
                                handler.LoggingHandler(['val'], id='logging_val')
                            ]
                        )
                    ]
                )
            ]
        )

    def build_eval(self, ctx: "Context"):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build evaluating process using handlers
        ctx.pipeline_ctx.eval_container = handler.RootContainer(
            id='container',
            handlers=[
                handler.Container(
                    id='container_eval',
                    wrappers=[
                        handler.StateWrapper(id='state_eval', state='eval')
                    ],
                    handlers=[
                        # clear meter metrics
                        handler.MeterInitHandler(id='meter_init_eval'),
                        # dataset iteration
                        handler.IterationContainer(
                            id='iteration_eval',
                            handlers=[
                                # forward
                                handler.ForwardHandler(id='forward_eval'),
                                # compute loss
                                handler.LossHandler(id='loss_eval'),
                                # compute metrics
                                handler.MetricHandler(id='metrics_eval'),
                                # compute meter metrics
                                handler.MeterHandler(id='meter_eval')
                            ]
                        ),
                        # logging
                        handler.LoggingHandler(['eval'], id='logging_eval')
                    ]
                )
            ]
        )
    
    def build_predict(self, ctx: "Context"):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build predicting process using handlers
        ctx.pipeline_ctx.predict_container = handler.RootContainer(
            id='container',
            handlers=[
                handler.Container(
                    id='container_predict',
                    wrappers=[
                        handler.StateWrapper(id='state_predict', state='predict')
                    ],
                    handlers=[
                        # dataset iteration
                        handler.IterationContainer(
                            id='iteration_predict',
                            handlers=[
                                # forward
                                handler.ForwardHandler(id='forward_predict')
                            ]
                        )
                    ]
                )
            ]
        )
    
    def __str__(self) -> str:
        return 'Epoch'


@build_registry(key='step')
class StepBuild(VanillaBuild):

    def build_train(self, ctx: "Context"):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.pipeline_ctx.train_container = handler.RootContainer(
            id='container',
            handlers=[
                # train
                handler.Container(
                    id='container_train',
                    wrappers=[
                        handler.StateWrapper(id='state_train', state='train')
                    ],
                    handlers=[
                        # init meter setting
                        handler.MeterInitHandler(id='meter_init_train'),
                        # dataset iter
                        handler.StepIterationContainer(
                            id='step_iteration_train',
                            handlers=[
                                # forward
                                handler.ForwardHandler(id='forward_train'),
                                # compute loss
                                handler.LossHandler(id='loss_train'),
                                # backward and optimizer step
                                handler.OptimizerContainer(
                                    id='optimizer_train',
                                    handlers=[
                                        handler.BackwardHandler(id='backward_train')
                                    ]
                                ),
                                # compute metrics
                                handler.MetricHandler(id='metrics_train'),
                                # compute meter loss value and metrics
                                handler.MeterHandler(id='meter_train'),
                                # apply learning rate schedule
                                handler.LRScheduleHandler(id='lr_schedule'),
                                # validation
                                handler.Container(
                                    id='container_val',
                                    wrappers=[
                                        handler.ConditionWrapper(id='condition_val', condition=validation_check),
                                        handler.StateWrapper(id='state_val', state='val')
                                    ],
                                    handlers=[
                                        handler.FuncHandler(
                                            id='logging_val_start',
                                            exec_ranks=[0],
                                            func_list=[
                                                lambda _: logger.info('Validation starts.')
                                            ]
                                        ),
                                        # init meter setting
                                        handler.MeterInitHandler(id='meter_init_val'),
                                        # dataset iter
                                        handler.IterationContainer(
                                            id='iteration_val',
                                            handlers=[
                                                # forward
                                                handler.ForwardHandler(id='forward_val'),
                                                # compute loss
                                                handler.LossHandler(id='loss_val'),
                                                # metrics
                                                handler.MetricHandler(id='metrics_val'),
                                                # compute meter loss value and metrics
                                                handler.MeterHandler(id='meter_val')
                                            ]
                                        ),
                                        # logging
                                        handler.LoggingHandler(['train', 'val'], id='logging_train_val'),
                                        # init train meter after validation
                                        handler.MeterInitHandler(
                                            id='meter_init_train_after_val',
                                            wrappers=[
                                                # set state to 'train' in order to init train metrics
                                                handler.StateWrapper(id='state_train_for_init', state='train')
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    
    def __str__(self) -> str:
        return 'Step'
