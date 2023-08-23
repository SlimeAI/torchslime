from torchslime.core.context import BaseContext
from torchslime.components.registry import Registry
from torchslime.core.handlers.wrappers import validation_check
from torchslime.core.handlers import ID, Wrappers
from torchslime.log import logger

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
        ctx.run_ctx.train = handler.Container[ID('container')]([
            # epoch iter
            handler.EpochIteration[ID('epoch_iteration')]([
                # train part
                handler.Container[
                    ID('container_train'),
                    Wrappers(
                        # set state to train
                        handler.State[ID('state_train')](state='train')
                    )
                ]([
                    # init meter setting
                    handler.MeterInit[ID('meter_init_train')](),
                    # dataset iter
                    handler.Iteration[ID('iteration_train')]([
                        # forward
                        handler.Forward[ID('forward_train')](),
                        # compute loss
                        handler.Loss[ID('loss_train')](),
                        # backward and optimizer step
                        handler.Optimizer[ID('optimizer_train')]([
                            handler.Backward[ID('backward_train')]()
                        ]),
                        # compute metrics
                        handler.Metrics[ID('metrics_train')](),
                        # compute meter loss value and metrics
                        handler.Meter[ID('meter_train')](),
                        # apply learning rate decay
                        handler.LRDecay[ID('lr_decay')](),
                        # display in console or in log files
                        handler.Display[ID('display_train')]()
                    ])
                ]),
                # validation part
                handler.Container[
                    ID('container_val'),
                    Wrappers(
                        # validation according to valid_seq
                        handler.Condition[ID('condition_val')](condition=validation_check),
                        # set state to val
                        handler.State[ID('state_val')](state='val')
                    )
                ]([
                    handler.Lambda[ID('print_val_start')]([lambda _: logger.info('Validation starts.')]),
                    # init meter setting
                    handler.MeterInit[ID('meter_init_val')](),
                    # dataset iter
                    handler.Iteration[ID('iteration_val')]([
                        # forward
                        handler.Forward[ID('forward_val')](),
                        # compute loss
                        handler.Loss[ID('loss_val')](),
                        # metrics
                        handler.Metrics[ID('metrics_val')](),
                        # compute meter loss value and metrics
                        handler.Meter[ID('meter_val')](),
                        # display in console or in log files
                        handler.Display[ID('display_val')]()
                    ])
                ])
            ])
        ])
        
        # TODO: decay mode

    def build_eval(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build evaluating process using handlers
        ctx.run_ctx.eval = handler.Container[ID('container')]([
            handler.Container[
                ID('container_eval'),
                Wrappers(
                    handler.State[ID('state_eval')](state='eval')
                )
            ]([
                # clear meter metrics
                handler.MeterInit[ID('meter_init_eval')](),
                # dataset iteration
                handler.Iteration[ID('iteration_eval')]([
                    # forward
                    handler.Forward[ID('forward_eval')](),
                    # compute loss
                    handler.Loss[ID('loss_eval')](),
                    # compute metrics
                    handler.Metrics[ID('metrics_eval')](),
                    # compute meter metrics
                    handler.Meter[ID('meter_eval')](),
                    # display
                    handler.Display[ID('display_eval')]()
                ])
            ])
        ])
    
    def build_predict(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build predicting process using handlers
        ctx.run_ctx.predict = handler.Container[ID('container')]([
            handler.Container[
                ID('container_predict'),
                Wrappers(
                    handler.State[ID('state_predict')](state='predict')
                )
            ]([
                # dataset iteration
                handler.Iteration[ID('iteration_predict')]([
                    # forward
                    handler.Forward[ID('forward_predict')](),
                    # display
                    handler.Display[ID('display_predict')]()
                ])
            ])
        ])


@build_registry(name='step')
class StepBuild(VanillaBuild):

    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train = handler.Container[ID('container')]([
            # train
            handler.Container[
                ID('container_train'),
                Wrappers(
                    handler.State[ID('state_train')](state='train')
                )
            ]([
                # init meter setting
                handler.MeterInit[ID('meter_init_train')](),
                # dataset iter
                handler.StepIteration[ID('step_iteration_train')]([
                    # forward
                    handler.Forward[ID('forward_train')](),
                    # compute loss
                    handler.Loss[ID('loss_train')](),
                    # backward and optimizer step
                    handler.Optimizer[ID('optimizer_train')]([
                        handler.Backward[ID('backward_train')]()
                    ]),
                    # compute metrics
                    handler.Metrics[ID('metrics_train')](),
                    # compute meter loss value and metrics
                    handler.Meter[ID('meter_train')](),
                    # apply learning rate decay
                    handler.LRDecay[ID('lr_decay')](),
                    # display in console or in log files
                    handler.Display[ID('display_train')](),
                    # validation
                    handler.Container[
                        ID('container_val'),
                        Wrappers(
                            handler.Condition[ID('condition_val')](condition=validation_check),
                            handler.State[ID('state_val')](state='val')
                        )
                    ]([
                        handler.Lambda[ID('print_val_start')]([lambda _: logger.info('Validation starts.')]),
                        # init meter setting
                        handler.MeterInit[ID('meter_init_val')](),
                        # dataset iter
                        handler.Iteration[ID('iteration_val')]([
                            # forward
                            handler.Forward[ID('forward_val')](),
                            # compute loss
                            handler.Loss[ID('loss_val')](),
                            # metrics
                            handler.Metrics[ID('metrics_val')](),
                            # compute meter loss value and metrics
                            handler.Meter[ID('meter_val')](),
                            # display in console or in log files
                            handler.Display[ID('display_val')]()
                        ]),
                        # init train meter after validation
                        handler.MeterInit[
                            ID('meter_init_train_after_val'),
                            Wrappers(
                                # set state to 'train' in order to init train metrics
                                handler.State[ID('state_train_for_init')](state='train')
                            )
                        ]()
                    ])
                ])
            ])
        ])
