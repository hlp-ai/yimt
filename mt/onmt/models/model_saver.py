import os
import torch
import re
from collections import deque
from onmt.utils.logging import logger
from onmt.inputters.inputter import vocabs_to_dict


def build_model_saver(model_opt, opt, model, vocabs, optim, device_id):
    # _check_save_model_path
    save_model_path = os.path.abspath(opt.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opt.save_model,
        model,
        model_opt,
        vocabs,
        optim,
        opt.keep_checkpoint,
        opt.save_format,
        device_id,
    )
    return model_saver


def load_checkpoint(ckpt_path):
    """Load checkpoint from `ckpt_path` if any else return `None`."""
    checkpoint = None
    if ckpt_path:
        logger.info("Loading checkpoint from %s" % ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        if "model" in checkpoint.keys():
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s)
                s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s)
                return s

            checkpoint["model"] = {fix_key(k): v for k, v in checkpoint["model"].items()}
            # Force add_ffnbias to True if bias found in model w_1 keys
            for key in checkpoint["model"].keys():
                if "w_1.bias" in key:
                    checkpoint["opt"].add_ffnbias = True

        if not hasattr(checkpoint["opt"], "num_kv"):
            checkpoint["opt"].num_kv = 0
        if not hasattr(checkpoint["opt"], "add_ffnbias"):
            checkpoint["opt"].add_ffnbias = False
        if not hasattr(checkpoint["opt"], "use_ckpting"):
            checkpoint["opt"].use_ckpting = []
        if not hasattr(checkpoint["opt"], "relative_positions_buckets"):
            checkpoint["opt"].relative_positions_buckets = 0
        if not hasattr(checkpoint["opt"], "parallel_mode"):
            checkpoint["opt"].parallel_mode = "data_parallel"
        if not hasattr(checkpoint["opt"], "norm_eps"):
            checkpoint["opt"].norm_eps = 1e-6

    return checkpoint


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(
        self,
        base_path,
        model,
        model_opt,
        vocabs,
        optim,
        keep_checkpoint=-1,
        save_format="pytorch",
        device_id=0,
    ):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.vocabs = vocabs
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        self.save_format = save_format
        self.device_id = device_id

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def save(self, step):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        ckpt_path, _ = self._save(step, save_model)

        self.last_saved_step = step

        if ckpt_path is not None:  # not None when process id 0
            if self.keep_checkpoint > 0:
                if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                    todel = self.checkpoint_queue.popleft()
                    self._rm_checkpoint(todel)

                self.checkpoint_queue.append(ckpt_path)

    def _save(self, step, model):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            model (nn.Module): torch model to save

        Returns:
            (str, str):

            * checkpoint_name: name (or path) of the saved checkpoint
            * model_name: name (or path) of the saved safetensors weights if applicable
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if "generator" not in k}
        generator_state_dict = model.generator.state_dict()

        if torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
        else:
            ws = 1
        if ws > 1:
            full_model = [None for _ in range(ws)]
            for key, value in model_state_dict.items():
                model_state_dict[key] = value.cpu()
            torch.distributed.all_gather_object(full_model, model_state_dict)
            fm_sd = {}
            for key in full_model[0].keys():
                if key.split(".")[-1] in [
                    "linear_keys",
                    "linear_values",
                    "linear_query",
                    "w_1",
                    "w_3",
                ]:
                    fm_sd[key] = torch.cat([full_model[i][key].cpu() for i in range(ws)], 0)
                elif key.split(".")[-1] in ["final_linear", "w_2"]:
                    fm_sd[key] = torch.cat([full_model[i][key].cpu() for i in range(ws)], 1)
                else:
                    fm_sd[key] = full_model[0][key]
            model_state_dict = fm_sd

        checkpoint = {
            "model": model_state_dict,
            "generator": generator_state_dict,
            "vocab": vocabs_to_dict(self.vocabs),
            "opt": self.model_opt,
            "optim": self.optim.state_dict(),
        }
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
            ckpt_path = "%s_step_%d.pt" % (self.base_path, step)
            torch.save(checkpoint, ckpt_path)
        else:
            ckpt_path = None
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return ckpt_path, None

    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)
