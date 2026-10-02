import argparse
import torch

@torch.inference_mode()
def generate_images(
    args: argparse.Namespace,
    generator: torch.nn.Module,
    labels: list[int] | torch.Tensor,
    tokenizer: torch.nn.Module | None = None,
    cfg: float = 1.0,
    z_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a batch of images. Returns NCHW float tensor in [0, 1].

    If *z_t* is provided it is forwarded to the model's ``generate`` method
    (the model must accept ``z_t`` as a keyword argument, e.g. pMFDenoiser).
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long).to("cuda")
    generator = generator.eval().to("cuda")
    with torch.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
        gen_kwargs = dict(
            n_samples=len(labels),
            cfg=cfg,
            labels=labels,
            args=args,
            verbose=args.num_sampling_steps > 2,
        )
        if z_t is not None:
            gen_kwargs["z_t"] = z_t
        generated = generator.generate(**gen_kwargs)
        if tokenizer is not None:
            generated = tokenizer.detokenize(generated)
        else:
            generated = ((generated + 1) / 2).clamp(0, 1)
    return generated