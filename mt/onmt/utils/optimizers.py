""" Optimizers class """
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import functools
from math import sqrt
import types
import os
import importlib
from onmt.utils.misc import fn_args

try:
    import apex
except ImportError:
    pass


def build_torch_optimizer(model, opt):
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well

    Args:
      model: The model to optimize.
      opt. The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = [opt.adam_beta1, opt.adam_beta2]
    if opt.optim == "sgd":
        optimizer = optim.SGD(params, lr=opt.learning_rate)
    elif opt.optim == "adagrad":
        optimizer = optim.Adagrad(
            params,
            lr=opt.learning_rate,
            initial_accumulator_value=opt.adagrad_accumulator_init,
        )
    elif opt.optim == "adadelta":
        optimizer = optim.Adadelta(params, lr=opt.learning_rate)
    elif opt.optim == "adam":
        optimizer = optim.Adam(params, lr=opt.learning_rate, betas=betas, eps=1e-8)
    elif opt.optim == "sparseadam":
        dense = []
        sparse = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # TODO: Find a better way to check for sparse gradients.
            if "embed" in name:
                sparse.append(param)
            else:
                dense.append(param)
        optimizer = MultipleOptimizer(
            [
                optim.Adam(dense, lr=opt.learning_rate, betas=betas, eps=1e-8),
                optim.SparseAdam(sparse, lr=opt.learning_rate, betas=betas, eps=1e-8),
            ]
        )
    elif opt.optim == "fusedadam":
        optimizer = FusedAdam(params, lr=opt.learning_rate, betas=betas)
        try:
            import apex
        except ImportError:
            raise ImportError("Could not import apex")
        if opt.apex_opt_level in ["O0", "O1", "O2", "O3"]:
            # we use apex.amp
            loss_scale = "dynamic" if opt.loss_scale == 0 else opt.loss_scale
            model, optimizer = apex.amp.initialize(
                [model, model.generator],
                optimizer,
                opt_level=opt.apex_opt_level,
                loss_scale=loss_scale,
                keep_batchnorm_fp32=None,
            )
        else:
            if opt.model_dtype == "fp16":
                # In this case use the old FusedAdam with
                # FP16_optimizer wrapper
                static_loss_scale = opt.loss_scale
                dynamic_loss_scale = opt.loss_scale == 0
                optimizer = apex.contrib.optimizers.FP16_Optimizer(
                    optimizer,
                    static_loss_scale=static_loss_scale,
                    dynamic_loss_scale=dynamic_loss_scale,
                )
    else:
        raise ValueError("Invalid optimizer type: " + opt.optim)

    return optimizer


def make_learning_rate_decay_fn(opt):
    """Returns the learning decay function from options."""
    if opt.decay_method == "noam":
        return functools.partial(
            noam_decay, warmup_steps=opt.warmup_steps, model_size=opt.hidden_size
        )
    elif opt.decay_method == "noamwd":
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.hidden_size,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )
    elif opt.decay_method == "rsqrt":
        return functools.partial(rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (
        model_size ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        * rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    )


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


class MultipleOptimizer(object):
    """Implement multiple optimizers needed for sparse adam"""

    def __init__(self, op):
        """?"""
        self.optimizers = op

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self, set_to_none=True):
        """?"""
        for op in self.optimizers:
            op.zero_grad(set_to_none)

    def step(self):
        """?"""
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """?"""
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """?"""
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """?"""
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
        optimizer: A ``torch.optim.Optimizer`` instance.
        learning_rate: The initial learning rate.
        learning_rate_decay_fn: An optional callable taking the current step
            as argument and return a learning rate scaling factor.
        max_grad_norm: Clip gradients to this global norm.
    """

    def __init__(
        self, optimizer, learning_rate, learning_rate_decay_fn=None, max_grad_norm=None
    ):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self._fp16 = None
        self._scaler = None

    @classmethod
    def from_opt(cls, model, opt, checkpoint=None):
        """Builds the optimizer from options.

        Args:
          cls: The ``Optimizer`` class to instantiate.
          model: The model to optimize.
          opt: The dict of user options.
          checkpoint: An optional checkpoint to load states from.

        Returns:
          An ``Optimizer`` instance.
        """
        optim_opt = opt
        optim_state_dict = None

        if opt.train_from and checkpoint is not None and "optim" in checkpoint.keys():
            optim = checkpoint["optim"]
            ckpt_opt = checkpoint["opt"]
            ckpt_state_dict = {}
            if isinstance(optim, Optimizer):  # Backward compatibility.
                ckpt_state_dict["training_step"] = optim._step + 1
                ckpt_state_dict["decay_step"] = optim._step + 1
                ckpt_state_dict["optimizer"] = optim.optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            if opt.reset_optim == "none":
                # Load everything from the checkpoint.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
            elif opt.reset_optim == "all":
                # Build everything from scratch.
                pass
            elif opt.reset_optim == "states":
                # Reset optimizer, keep options.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
                del optim_state_dict["optimizer"]
            elif opt.reset_optim == "keep_states":
                # Reset options, keep optimizer.
                optim_state_dict = ckpt_state_dict

        optimizer = cls(
            build_torch_optimizer(model, optim_opt),
            optim_opt.learning_rate,
            learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt),
            max_grad_norm=optim_opt.max_grad_norm,
        )
        if opt.model_dtype == "fp16":
            if opt.optim == "fusedadam":
                if opt.apex_opt_level in ["O0", "O1", "O2", "O3"]:
                    optimizer._fp16 = "apex.amp"
                else:
                    optimizer._fp16 = "legacy"
            else:
                optimizer._fp16 = "amp"
                from torch.cuda.amp import GradScaler

                optimizer._scaler = GradScaler()
        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)
        return optimizer

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    @property
    def amp(self):
        """True if use torch amp mix precision training."""
        return self._fp16 == "amp"

    def learning_rate(self):
        """Returns the current learning rate."""
        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        scale = self._learning_rate_decay_fn(self._decay_step)
        return scale * self._learning_rate

    def state_dict(self):
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict["training_step"]
        # State can be partially restored.
        if "decay_step" in state_dict:
            self._decay_step = state_dict["decay_step"]
        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self, set_to_none=True):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad()
        # should be: self._optimizer.zero_grad(set_to_none)
        # but apex.amp is not up-to-date:
        # https://github.com/NVIDIA/apex/blob/master/apex/amp/_process_optimizer.py#L367

    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the
        backward pass."""
        if self._fp16 == "legacy":
            kwargs = {}
            if "update_master_grads" in fn_args(self._optimizer.backward):
                kwargs["update_master_grads"] = True
            self._optimizer.backward(loss, **kwargs)
        elif self.amp:
            self._scaler.scale(loss).backward()
        elif self._fp16 == "apex.amp":
            with apex.amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        learning_rate = self.learning_rate()

        if self.amp:
            self._scaler.unscale_(self._optimizer)
        elif self._fp16 == "legacy":
            if hasattr(self._optimizer, "update_master_grads"):
                self._optimizer.update_master_grads()
            if (
                hasattr(self._optimizer, "clip_master_grads")
                and self._max_grad_norm > 0
            ):
                self._optimizer.clip_master_grads(self._max_grad_norm)

        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if self._max_grad_norm > 0 and self._fp16 != "legacy":
                clip_grad_norm_(group["params"], self._max_grad_norm)

        if self.amp:
            # unscaled optimizer's gradients (already done therefore skip),
            # skips optimizer.step() if gradients contain infs/NaNs.
            self._scaler.step(self._optimizer)
            # Updates the scale for next iteration.
            self._scaler.update()
        else:
            self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1


class FusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.
       Requires Apex to be installed via
       ``python setup.py install --cuda_ext --cpp_ext``.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper 'On the Convergence of Adam and Beyond'
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        amsgrad=False,
    ):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError("AMSGrad variant not supported.")
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(
        self, closure=None, grads=None, output_params=None, scale=1.0, grad_norms=None
    ):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients.
                (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        for group, grads_this_group, output_params_this_group, grad_norm in zip(
            self.param_groups, grads_group, output_params_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group["params"])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group["params"])

            # compute combined scale factor for this group
            combined_scale = scale
            if group["max_grad_norm"] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group["max_grad_norm"]
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group["bias_correction"] else 0

            for p, grad, output_param in zip(
                group["params"], grads_this_group, output_params_this_group
            ):
                # note: p.grad should not ever be set for correct operation of
                # mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse \
                                       gradients, please consider \
                                       SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = (
                    torch.tensor([], dtype=torch.float)
                    if output_param is None
                    else output_param
                )
                fused_adam_cuda.adam(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    combined_scale,
                    state["step"],
                    self.eps_mode,
                    bias_correction,
                    group["weight_decay"],
                )
        return loss
