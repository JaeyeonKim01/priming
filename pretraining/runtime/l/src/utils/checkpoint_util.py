import datetime
import logging
import os
import threading
from glob import glob

import torch

import src.utils.distributed_util as dist


logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(x) for x in obj)
    return obj


def _load_model_sd(model, state_dict, strict=True):
    """Load state_dict into model."""
    return model.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# async checkpoint saver
# ---------------------------------------------------------------------------

class AsyncCheckpointSaver:
    """Write checkpoints to disk in a background thread.

    Usage::

        saver = AsyncCheckpointSaver()
        ...
        saver.save(checkpoint_cpu, path, post_save_fn)   # returns immediately
        ...
        saver.wait()   # blocks until save is done (call before next save or at exit)
    """

    def __init__(self):
        self._thread = None
        self._error = None

    def wait(self):
        """Block until the most recent save completes (no-op if idle)."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

    def save(self, checkpoint, path, post_save_fn=None):
        """Enqueue *checkpoint* (already on CPU) for background writing."""
        self.wait()  # serialise: only one save at a time

        def _worker():
            try:
                temporary = path + ".tmp"
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
                logger.info(f"[async] Saved checkpoint: {path}")
                if post_save_fn is not None:
                    post_save_fn()
            except Exception as e:
                logger.error(f"[async] Checkpoint save failed: {e}")
                self._error = e

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()


# ---------------------------------------------------------------------------
# checkpoint load / resume
# ---------------------------------------------------------------------------

def _load_ema_from_checkpoint(
    model_ema,
    model,
    checkpoint,
    prefix="Model-resume",
):
    """load ema weights from a checkpoint into model_ema, handling all legacy formats."""
    if model_ema is None:
        return

    if "model_ema" in checkpoint:
        logger.info(f"[{prefix}] Loading EMA from 'model_ema'")
        model_ema.load_state_dict(checkpoint["model_ema"])
    elif "model_ema1" in checkpoint and "model_ema2" in checkpoint:
        logger.info(f"[{prefix}] Loading EMA from 'model_ema1' + 'model_ema2' (JiT format)")
        model_ema.load_state_dict(checkpoint["model_ema1"], label=model_ema.labels[0])
        logger.info(f"[{prefix}] model_ema1 -> label='{model_ema.labels[0]}'")
        for label in model_ema.labels[1:]:
            model_ema.load_state_dict(checkpoint["model_ema2"], label=label)
            logger.info(f"[{prefix}] model_ema2 -> label='{label}'")
    else:
        logger.info(f"[{prefix}] No EMA in checkpoint, initializing from model weights")
        param_keys = [k for k, _ in model.named_parameters()]
        model_sd = model.state_dict()
        ema_sd = {k: model_sd[k] for k in param_keys if k in model_sd}
        for label in model_ema.labels:
            model_ema.load_state_dict(ema_sd, label=label)

    model_ema.to("cuda")


def ckpt_resume(
    args,
    model,
    optimizer=None,
    model_ema=None,
    extra_keys=None,
):
    """Resume from checkpoint.

    When *extra_keys* is given (list of strings), those keys are extracted
    from the checkpoint and returned as a dict.  This is used e.g. to
    restore FD queue states without re-generating 50 k images.

    Returns
    -------
    dict or None
        Extracted extra data if *extra_keys* was provided and a checkpoint
        was loaded, else ``None``.
    """
    extra_data = None

    if args.resume_from or args.auto_resume:
        if args.resume_from is None:
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            msg = _load_model_sd(model, checkpoint["model"])
            logger.info(f"[Model-resume] Loaded model: {msg}")

            _load_ema_from_checkpoint(model_ema, model, checkpoint, prefix="Model-resume")

            if "optimizer" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"[Model-resume] Loaded optimizer: {optimizer}")

            if "last_elapsed_time" in checkpoint:
                args.last_elapsed_time = float(checkpoint["last_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.last_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")

            if "step" in checkpoint:
                args.current_step = int(checkpoint["step"]) + 1
                logger.info(f"Loaded current_step: {args.current_step}")
            if "samples_seen" in checkpoint:
                args.samples_seen = int(checkpoint["samples_seen"])
                logger.info(f"Loaded samples_seen: {args.samples_seen}")

            args.start_epoch = max(args.current_step - 1, 0) // args.steps_per_epoch

            if extra_keys:
                extra_data = {k: checkpoint[k] for k in extra_keys if k in checkpoint}
                if extra_data:
                    logger.info(f"[Model-resume] Extracted extra keys: {list(extra_data.keys())}")

            del checkpoint
        else:
            logger.info(f"[Model-resume] Could not find checkpoint at {args.resume_from}.")

    if args.load_from and not args.resume_from:
        if os.path.exists(args.load_from):
            import src.models as models
            logger.info(f"[Model-load] Loading checkpoint from: {args.load_from}")
            checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            if args.model in models.iMFDenoiser_models:
                from src.models.denoiser_imf import convert_imf_checkpoint
                logger.info(f"[Model-load] Converting official iMF checkpoint keys")
                state_dict = convert_imf_checkpoint(state_dict)

            if args.model in models.pMFDenoiser_models:
                from src.models.denoiser_pmf import convert_pmf_checkpoint
                logger.info(f"[Model-load] Converting official pMF checkpoint keys")
                state_dict = convert_pmf_checkpoint(state_dict)

            if len(state_dict) > 0:
                msg = _load_model_sd(model, state_dict, strict=False)
                logger.info(f"[Model-load] Loaded model: {msg}")

            _load_ema_from_checkpoint(model_ema, model, checkpoint, prefix="Model-load")
            del checkpoint
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {args.load_from}")
    return extra_data


def cleanup_checkpoints(ckpt_dir, keep_num=5, milestone_every=50000):
    """Clean up older checkpoint files, keeping the latest keep_num + milestones."""
    ckpts = glob(os.path.join(ckpt_dir, "*.pth"))
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]

    def get_ckpt_step(path):
        filename = os.path.basename(path)
        try:
            return int(filename.rsplit("_", 1)[-1].split(".")[0])
        except ValueError:
            return None

    ckpts.sort(key=lambda x: (get_ckpt_step(x) is None, get_ckpt_step(x)))
    ckpts = [ckpt for ckpt in ckpts if get_ckpt_step(ckpt) is not None]

    if not ckpts:
        return

    newest_keep = set(ckpts[-keep_num:])
    milestone_keep = set(
        ckpt for ckpt in ckpts
        if milestone_every > 0 and get_ckpt_step(ckpt) % milestone_every == 0
    )

    keep_set = newest_keep.union(milestone_keep)

    for ckpt in ckpts:
        if ckpt not in keep_set:
            os.remove(ckpt)
            logger.info(f"Removed checkpoint: {ckpt}")

    # ckpts is already sorted — last kept entry is the newest
    newest_ckpt = os.path.abspath(ckpts[-1])
    latest_symlink = os.path.join(ckpt_dir, "latest.pth")

    try:
        os.remove(latest_symlink)
    except FileNotFoundError:
        pass

    os.symlink(newest_ckpt, latest_symlink)
    logger.info(f"Created symlink: {latest_symlink} -> {newest_ckpt}")


def save_checkpoint(
    args,
    step,
    model,
    optimizer,
    model_ema,
    elapsed_time=0.0,
    saver=None,
    extra=None,
):
    """Save a training checkpoint.

    If *saver* (an ``AsyncCheckpointSaver``) is provided, the state dicts are
    snapshotted to CPU immediately and the actual disk write happens in a
    background thread so training can resume without waiting for Lustre I/O.

    """
    model_sd = model.state_dict()

    if not dist.is_main_process():
        return
    checkpoint_data = {
        "model": model_sd,
        "model_ema": model_ema.state_dict() if model_ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "last_elapsed_time": elapsed_time,
        "current_step": args.current_step,
        "samples_seen": args.samples_seen,
    }
    if extra is not None:
        checkpoint_data.update(extra)
    checkpoint_path = os.path.join(args.ckpt_dir, f"step_{step:07d}.pth")

    if saver is not None:
        # snapshot to CPU so the background thread owns independent copies;
        # disk I/O then happens off the main thread
        checkpoint_data = _to_cpu(checkpoint_data)
        ckpt_dir = args.ckpt_dir
        keep_n = args.keep_n_ckpts
        milestone = args.milestone_every
        saver.save(
            checkpoint_data, checkpoint_path,
            post_save_fn=lambda: cleanup_checkpoints(ckpt_dir, keep_n, milestone),
        )
    else:
        # synchronous save (torch.save handles GPU tensors natively)
        temporary = checkpoint_path + ".tmp"
        torch.save(checkpoint_data, temporary)
        os.replace(temporary, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        cleanup_checkpoints(args.ckpt_dir, args.keep_n_ckpts, args.milestone_every)
