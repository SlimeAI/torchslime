from torchslime.core.context import BaseContext
from torchslime.components.registry import Registry
from torchslime.core.handlers.wrappers import validation_check
from torchslime.logging.logger import logger

build_registry = Registry('build_registry')


class BuildHook:

    def build_train(self, ctx: BaseContext) -> None: pass
    def build_eval(self, ctx: BaseContext) -> None: pass
    def build_predict(self, ctx: BaseContext) -> None: pass

    def _build_train(self, ctx: BaseContext):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx
        h.launch.before_build_train(ctx)
        h.plugins.before_build_train(ctx)
        h.build.build_train(ctx)
        h.launch.after_build_train(ctx)
        h.plugins.after_build_train(ctx)
    
    def _build_eval(self, ctx: BaseContext):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx
        h.launch.before_build_eval(ctx)
        h.plugins.before_build_eval(ctx)
        h.build.build_eval(ctx)
        h.launch.after_build_eval(ctx)
        h.plugins.after_build_eval(ctx)

    def _build_predict(self, ctx: BaseContext):
        """
        Build order:
        Launch -> Plugin -> Build -> Launch -> Plugin
        """
        h = ctx.hook_ctx
        h.launch.before_build_predict(ctx)
        h.plugins.before_build_predict(ctx)
        h.build.build_predict(ctx)
        h.launch.after_build_predict(ctx)
        h.plugins.after_build_predict(ctx)


class _BuildInterface:
    """
    Interface for building handlers.
    """
    def before_build_train(self, ctx: BaseContext) -> None: pass
    def after_build_train(self, ctx: BaseContext) -> None: pass
    def before_build_eval(self, ctx: BaseContext) -> None: pass
    def after_build_eval(self, ctx: BaseContext) -> None: pass
    def before_build_predict(self, ctx: BaseContext) -> None: pass
    def after_build_predict(self, ctx: BaseContext) -> None: pass


@build_registry(name='vanilla')
class VanillaBuild(BuildHook):
    
    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train_container = handler.RootContainer.m__(id='container')([
            # epoch iter
            handler.EpochIterationContainer.m__(id='epoch_iteration')([
                # train part
                handler.Container.m__(
                    id='container_train',
                    wrappers=[
                        # set state to train
                        handler.StateWrapper.m__(id='state_train')(state='train')
                    ]
                )([
                    # init meter setting
                    handler.MeterInitHandler.m__(id='meter_init_train')(),
                    # dataset iter
                    handler.IterationContainer.m__(id='iteration_train')([
                        # forward
                        handler.ForwardHandler.m__(id='forward_train')(),
                        # compute loss
                        handler.LossHandler.m__(id='loss_train')(),
                        # backward and optimizer step
                        handler.OptimizerContainer.m__(id='optimizer_train')([
                            handler.BackwardHandler.m__(id='backward_train')()
                        ]),
                        # compute metrics
                        handler.MetricHandler.m__(id='metrics_train')(),
                        # compute meter loss value and metrics
                        handler.MeterHandler.m__(id='meter_train')(),
                        # apply learning rate decay
                        handler.LRDecayHandler.m__(id='lr_decay')()
                    ]),
                    # logging
                    handler.LoggingHandler.m__(id='logging_train')(['train'])
                ]),
                # validation part
                handler.Container.m__(
                    id='container_val',
                    wrappers=[
                        # validation according to valid_seq
                        handler.ConditionWrapper.m__(id='condition_val')(condition=validation_check),
                        # set state to val
                        handler.StateWrapper.m__(id='state_val')(state='val')
                    ]
                )([
                    handler.FuncHandler.m__(
                        id='logging_val_start',
                        exec_ranks=[0]
                    )([lambda _: logger.info('Validation starts.')]),
                    # init meter setting
                    handler.MeterInitHandler.m__(id='meter_init_val')(),
                    # dataset iter
                    handler.IterationContainer.m__(id='iteration_val')([
                        # forward
                        handler.ForwardHandler.m__(id='forward_val')(),
                        # compute loss
                        handler.LossHandler.m__(id='loss_val')(),
                        # metrics
                        handler.MetricHandler.m__(id='metrics_val')(),
                        # compute meter loss value and metrics
                        handler.MeterHandler.m__(id='meter_val')()
                    ]),
                    # logging
                    handler.LoggingHandler.m__(id='logging_val')(['val'])
                ])
            ])
        ])

    def build_eval(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build evaluating process using handlers
        ctx.run_ctx.eval_container = handler.RootContainer.m__(id='container')([
            handler.Container.m__(
                id='container_eval',
                wrappers=[
                    handler.StateWrapper.m__(id='state_eval')(state='eval')
                ]
            )([
                # clear meter metrics
                handler.MeterInitHandler.m__(id='meter_init_eval')(),
                # dataset iteration
                handler.IterationContainer.m__(id='iteration_eval')([
                    # forward
                    handler.ForwardHandler.m__(id='forward_eval')(),
                    # compute loss
                    handler.LossHandler.m__(id='loss_eval')(),
                    # compute metrics
                    handler.MetricHandler.m__(id='metrics_eval')(),
                    # compute meter metrics
                    handler.MeterHandler.m__(id='meter_eval')()
                ]),
                # logging
                handler.LoggingHandler.m__(id='logging_eval')(['eval'])
            ])
        ])
    
    def build_predict(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build predicting process using handlers
        ctx.run_ctx.predict_container = handler.RootContainer.m__(id='container')([
            handler.Container.m__(
                id='container_predict',
                wrappers=[
                    handler.StateWrapper.m__(id='state_predict')(state='predict')
                ]
            )([
                # dataset iteration
                handler.IterationContainer.m__(id='iteration_predict')([
                    # forward
                    handler.ForwardHandler.m__(id='forward_predict')()
                ])
            ])
        ])
    
    def __str__(self) -> str:
        return 'Epoch'


@build_registry(name='step')
class StepBuild(VanillaBuild):

    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train_container = handler.RootContainer.m__(id='container')([
            # train
            handler.Container.m__(
                id='container_train',
                wrappers=[
                    handler.StateWrapper.m__(id='state_train')(state='train')
                ]
            )([
                # init meter setting
                handler.MeterInitHandler.m__(id='meter_init_train')(),
                # dataset iter
                handler.StepIterationContainer.m__(id='step_iteration_train')([
                    # forward
                    handler.ForwardHandler.m__(id='forward_train')(),
                    # compute loss
                    handler.LossHandler.m__(id='loss_train')(),
                    # backward and optimizer step
                    handler.OptimizerContainer.m__(id='optimizer_train')([
                        handler.BackwardHandler.m__(id='backward_train')()
                    ]),
                    # compute metrics
                    handler.MetricHandler.m__(id='metrics_train')(),
                    # compute meter loss value and metrics
                    handler.MeterHandler.m__(id='meter_train')(),
                    # apply learning rate decay
                    handler.LRDecayHandler.m__(id='lr_decay')(),
                    # validation
                    handler.Container.m__(
                        id='container_val',
                        wrappers=[
                            handler.ConditionWrapper.m__(id='condition_val')(condition=validation_check),
                            handler.StateWrapper.m__(id='state_val')(state='val')
                        ]
                    )([
                        handler.FuncHandler.m__(
                            id='logging_val_start',
                            exec_ranks=[0]
                        )([lambda _: logger.info('Validation starts.')]),
                        # init meter setting
                        handler.MeterInitHandler.m__(id='meter_init_val')(),
                        # dataset iter
                        handler.IterationContainer.m__(id='iteration_val')([
                            # forward
                            handler.ForwardHandler.m__(id='forward_val')(),
                            # compute loss
                            handler.LossHandler.m__(id='loss_val')(),
                            # metrics
                            handler.MetricHandler.m__(id='metrics_val')(),
                            # compute meter loss value and metrics
                            handler.MeterHandler.m__(id='meter_val')()
                        ]),
                        # logging
                        handler.LoggingHandler.m__(id='logging_train_val')(['train', 'val']),
                        # init train meter after validation
                        handler.MeterInitHandler.m__(
                            id='meter_init_train_after_val',
                            wrappers=[
                                # set state to 'train' in order to init train metrics
                                handler.StateWrapper.m__(id='state_train_for_init')(state='train')
                            ]
                        )()
                    ])
                ])
            ])
        ])
    
    def __str__(self) -> str:
        return 'Step'
