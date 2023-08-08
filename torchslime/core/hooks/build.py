from torchslime.core.context import BaseContext
from torchslime.components.registry import Registry

build_registry = Registry('build_registry')


class BuildHook:

    def build_train(self, ctx: BaseContext): pass
    def build_eval(self, ctx: BaseContext): pass
    def build_predict(self, ctx: BaseContext): pass

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
            # set global step to 0
            handler.Lambda([
                lambda ctx: setattr(ctx.iteration_ctx, 'current_step', 0)
            ], _id='global_step_init'),
            # epoch iter
            handler.EpochIteration([
                handler.Wrapper([
                    # init average setting
                    handler.AverageInit(_id='average_init_train'),
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
                        # compute average loss value and metrics
                        handler.Average(_id='average_train'),
                        # display in console or in log files
                        handler.Display(_id='display_train')
                    ], _id='iteration_train'),
                    # apply learning rate decay
                    handler.LRDecay(_id='lr_decay')
                ], wrappers=[
                    # set state to 'train'
                    handler.State(state='train', _id='state_train')
                ], _id='wrapper_train'),
                handler.Wrapper([
                    # init average setting
                    handler.AverageInit(_id='average_init_val'),
                    # dataset iter
                    handler.Iteration([
                        # forward
                        handler.Forward(_id='forward_val'),
                        # compute loss
                        handler.Loss(_id='loss_val'),
                        # metrics
                        handler.Metrics(_id='metrics_val'),
                        # compute average loss value and metrics
                        handler.Average(_id='average_val'),
                        # display in console or in log files
                        handler.Display(_id='display_val')
                    ], _id='iteration_val')
                ], wrappers=[
                    # set state to 'val'
                    handler.State('val', _id='state_val')
                ], _id='wrapper_val')
            ], _id='epoch_iteration')
        ], _id='container')
        
        # TODO: decay mode

    def build_eval(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build evaluating process using handlers
        ctx.run_ctx.eval = handler.Container([
            # set global step to 0
            handler.Lambda([
                lambda ctx: setattr(ctx.iteration_ctx, 'current_step', 0)
            ], _id='global_step_init'),
            handler.Wrapper([
                # clear average metrics
                handler.AverageInit(_id='eval_average_init'),
                # dataset iteration
                handler.Iteration([
                    # forward
                    handler.Forward(_id='eval_forward'),
                    # compute loss
                    handler.Loss(_id='eval_loss'),
                    # compute metrics
                    handler.Metrics(_id='eval_metrics'),
                    # compute average metrics
                    handler.Average(_id='eval_average'),
                    # display
                    handler.Display(_id='eval_display')
                ], _id='eval_iteration')
            ], wrappers=[
                # set state to 'eval'
                handler.State('eval', _id='eval_state')
            ], _id='wrapper_eval')
        ], _id='eval_container')
    
    def build_predict(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build predicting process using handlers
        ctx.run_ctx.predict = handler.Container([
            # set global step to 0
            handler.Lambda([
                lambda ctx: setattr(ctx.iteration_ctx, 'current_step', 0)
            ], _id='global_step_init'),
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
                handler.State('predict', _id='predict_state')
            ], _id='wrapper_predict')
        ], _id='predict_container')


@build_registry.register(name='step')
class StepBuild(VanillaBuild):

    def build_train(self, ctx: BaseContext):
        # get handler classes from context
        handler = ctx.handler_ctx
        # build training process using handlers
        ctx.run_ctx.train = handler.Container([
            # set global step to 0
            handler.Lambda([
                lambda ctx: setattr(ctx.iteration_ctx, 'current_step', 0)
            ], _id='global_step_init'),
            handler.Wrapper([
                # init average setting
                handler.AverageInit(_id='average_init_train'),
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
                    # compute average loss value and metrics
                    handler.Average(_id='average_train'),
                    # display in console or in log files
                    handler.Display(_id='display_train'),
                    # validation
                    handler.Wrapper([
                        # init average setting
                        handler.AverageInit(_id='average_init_val'),
                        # dataset iter
                        handler.Iteration([
                            # forward
                            handler.Forward(_id='forward_val'),
                            # compute loss
                            handler.Loss(_id='loss_val'),
                            # metrics
                            handler.Metrics(_id='metrics_val'),
                            # compute average loss value and metrics
                            handler.Average(_id='average_val'),
                            # display in console or in log files
                            handler.Display(_id='display_val')
                        ], _id='iteration_val')
                    ], wrappers=[
                        # set state to 'val'
                        handler.State('val', _id='state_val')
                    ], _id='wrapper_val')
                ], _id='iteration_train'),
                # apply learning rate decay
                handler.LRDecay(_id='lr_decay')
            ], wrappers=[
                # set state to 'train'
                handler.State(state='train', _id='state_train')
            ], _id='wrapper_train')
        ], _id='container')
