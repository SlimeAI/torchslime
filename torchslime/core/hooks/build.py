from torchslime.core.context import BaseContext
from torchslime.components.registry import Registry
from torchslime.core.handlers.conditions import validation_check
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


@build_registry.register(name='vanilla')
class VanillaBuild(BuildHook):
    
    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train = handler.Container([
            # epoch iter
            handler.EpochIteration([
                handler.Wrapper([
                    # init meter setting
                    handler.MeterInit(_id='meter_init_train'),
                    # dataset iter
                    handler.Iteration([
                        # forward
                        handler.Forward(_id='forward_train'),
                        # compute loss
                        handler.Loss(_id='loss_train'),
                        # backward and optimizer step
                        handler.Optimizer([
                            handler.Backward(_id='backward_train')
                        ], _id='optimizer_train'),
                        # compute metrics
                        handler.Metrics(_id='metrics_train'),
                        # compute meter loss value and metrics
                        handler.Meter(_id='meter_train'),
                        # apply learning rate decay
                        handler.LRDecay(_id='lr_decay'),
                        # display in console or in log files
                        handler.Display(_id='display_train')
                    ], _id='iteration_train')
                ], wrappers=[
                    # set state to 'train'
                    handler.State(state='train', _id='state_train')
                ], _id='wrapper_train'),
                handler.Condition([
                    handler.Lambda([
                        lambda _: logger.info('\nValidation starts.')
                    ], _id='print_val_start'),
                    handler.Wrapper([
                        # init meter setting
                        handler.MeterInit(_id='meter_init_val'),
                        # dataset iter
                        handler.Iteration([
                            # forward
                            handler.Forward(_id='forward_val'),
                            # compute loss
                            handler.Loss(_id='loss_val'),
                            # metrics
                            handler.Metrics(_id='metrics_val'),
                            # compute meter loss value and metrics
                            handler.Meter(_id='meter_val'),
                            # display in console or in log files
                            handler.Display(_id='display_val')
                        ], _id='iteration_val')
                    ], wrappers=[
                        # set state to 'val'
                        handler.State(state='val', _id='state_val')
                    ], _id='wrapper_val')
                ], condition=validation_check)
            ], _id='epoch_iteration')
        ], _id='container')
        
        # TODO: decay mode

    def build_eval(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build evaluating process using handlers
        ctx.run_ctx.eval = handler.Container([
            handler.Wrapper([
                # clear meter metrics
                handler.MeterInit(_id='eval_meter_init'),
                # dataset iteration
                handler.Iteration([
                    # forward
                    handler.Forward(_id='eval_forward'),
                    # compute loss
                    handler.Loss(_id='eval_loss'),
                    # compute metrics
                    handler.Metrics(_id='eval_metrics'),
                    # compute meter metrics
                    handler.Meter(_id='eval_meter'),
                    # display
                    handler.Display(_id='eval_display')
                ], _id='eval_iteration')
            ], wrappers=[
                # set state to 'eval'
                handler.State(state='eval', _id='eval_state')
            ], _id='wrapper_eval')
        ], _id='eval_container')
    
    def build_predict(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build predicting process using handlers
        ctx.run_ctx.predict = handler.Container([
            handler.Wrapper([
                # dataset iteration
                handler.Iteration([
                    # forward
                    handler.Forward(_id='predict_forward'),
                    # display
                    handler.Display(_id='predict_display')
                ], _id='predict_iteration')
            ], wrappers=[
                # set state to 'predict'
                handler.State(state='predict', _id='predict_state')
            ], _id='wrapper_predict')
        ], _id='predict_container')


@build_registry.register(name='step')
class StepBuild(VanillaBuild):

    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train = handler.Container([
            # train
            handler.Wrapper([
                # init meter setting
                handler.MeterInit(_id='meter_init_train'),
                # dataset iter
                handler.StepIteration([
                    # forward
                    handler.Forward(_id='forward_train'),
                    # compute loss
                    handler.Loss(_id='loss_train'),
                    # backward and optimizer step
                    handler.Optimizer([
                        handler.Backward(_id='backward_train')
                    ], _id='optimizer_train'),
                    # compute metrics
                    handler.Metrics(_id='metrics_train'),
                    # compute meter loss value and metrics
                    handler.Meter(_id='meter_train'),
                    # apply learning rate decay
                    handler.LRDecay(_id='lr_decay'),
                    # display in console or in log files
                    handler.Display(_id='display_train'),
                    # validation
                    handler.Condition([
                        handler.Lambda([
                            lambda _: logger.info('\nValidation starts.')
                        ], _id='print_val_start'),
                        handler.Wrapper([
                            # init meter setting
                            handler.MeterInit(_id='meter_init_val'),
                            # dataset iter
                            handler.Iteration([
                                # forward
                                handler.Forward(_id='forward_val'),
                                # compute loss
                                handler.Loss(_id='loss_val'),
                                # metrics
                                handler.Metrics(_id='metrics_val'),
                                # compute meter loss value and metrics
                                handler.Meter(_id='meter_val'),
                                # display in console or in log files
                                handler.Display(_id='display_val')
                            ], _id='iteration_val')
                        ], wrappers=[
                            # set state to 'val'
                            handler.State(state='val', _id='state_val')
                        ], _id='wrapper_val'),
                        # init train meter after validation
                        handler.MeterInit(_id='meter_init_train_after_val')
                    ], condition=validation_check)
                ], _id='iteration_train')
            ], wrappers=[
                # set state to 'train'
                handler.State(state='train', _id='state_train')
            ], _id='wrapper_train')
        ], _id='container')
