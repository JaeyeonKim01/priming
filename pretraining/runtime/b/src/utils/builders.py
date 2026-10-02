import logging

import torch

import src.models as models
from src.utils.distributed_util import broadcast_module_params, is_enabled
from src.utils.ema_util import EMAModel

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# model / tokenizer creation
# ---------------------------------------------------------------------------

def create_generation_model(args):
    logger.info("Creating generation models.")


    if args.model in models.JiTDenoiser_models:
        model = models.JiTDenoiser_models[args.model](
            img_size=args.img_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            P_mean=args.P_mean,
            P_std=args.P_std,
            t_eps=args.t_eps,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            legacy_time_convention=args.legacy_time_convention,
        )
    elif args.model in models.MMJiTDenoiser_models:
        model = models.MMJiTDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            hidden_size=args.hidden_size,
            t5_hidden_size=args.t5_hidden_size,
            depth_double=args.depth_double,
            text_preamble_depth=args.text_preamble_depth,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            mlp_ratio=args.mlp_ratio,
            pca_channels=args.pca_channels,
            final_layer_zero=args.final_layer_zero,
            compat_checkpoint_arch=args.compat_checkpoint_arch,
            attention_impl=args.attention_impl,
            pretrained_checkpoint=args.pretrained_checkpoint,
            pretrained_source=args.pretrained_source,
            legacy_time_convention=args.legacy_time_convention,
            noise_scale=args.noise_scale,
            t_eps=args.t_eps,
        )
    elif args.model in models.iMFDenoiser_models:
        model = models.iMFDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.token_channels,
            tokenizer_patch_size=args.tokenizer_patch_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            # training parameters
            P_mean=args.P_mean,
            P_std=args.P_std,
            ratio_r_neq_t=args.ratio_r_neq_t,
            cfg_beta=args.cfg_beta,
            cfg_omega_max=args.cfg_omega_max,
            aux_head_depth=args.aux_head_depth,
            class_tokens=args.class_tokens,
            time_tokens=args.time_tokens,
            guidance_tokens=args.guidance_tokens,
            interval_tokens=args.interval_tokens,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            disable_v_head=args.disable_v_head,
        )
    elif args.model in models.pMFDenoiser_models:
        model = models.pMFDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.token_channels,
            tokenizer_patch_size=args.tokenizer_patch_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            # training parameters
            P_mean=args.P_mean,
            P_std=args.P_std,
            ratio_r_neq_t=args.ratio_r_neq_t,
            cfg_beta=args.cfg_beta,
            tr_uniform=args.tr_uniform,
            cfg_omega_max=args.cfg_omega_max,
            aux_head_depth=args.aux_head_depth,
            class_tokens=args.class_tokens,
            time_tokens=args.time_tokens,
            guidance_tokens=args.guidance_tokens,
            interval_tokens=args.interval_tokens,
            t_eps=args.t_eps,
            perceptual_threshold=args.perceptual_threshold,
            perceptual_loss_on_aux=args.perceptual_loss_on_aux,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            disable_v_head=args.disable_v_head,
            sampling_head=getattr(args, "sampling_head", "u"),
            freeze_inactive_head=getattr(args, "freeze_inactive_head", True),
            noise_scale=args.noise_scale,
            norm_eps=args.norm_eps,
            norm_p=args.norm_p,
            jvp_attention_backend=getattr(args, "jvp_attention_backend", "manual"),
        )
    elif args.model in models.OSTDenoiser_models:
        model = models.OSTDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.token_channels,
            tokenizer_patch_size=args.tokenizer_patch_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            corruption=getattr(args, "corruption", "denoising"),
            t_dist=getattr(args, "t_dist", "lognorm"),
            t_min=getattr(args, "t_min", 0.0),
            t_max=getattr(args, "t_max", 1.0),
            P_mean=args.P_mean,
            P_std=args.P_std,
            cfg_beta=args.cfg_beta,
            cfg_omega_max=args.cfg_omega_max,
            class_tokens=args.class_tokens,
            guidance_tokens=args.guidance_tokens,
            interval_tokens=args.interval_tokens,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            noise_scale=args.noise_scale,
            norm_eps=args.norm_eps,
            norm_p=args.norm_p,
            perceptual_threshold=args.perceptual_threshold,
        )
    elif args.model in models.RAEDenoiser_models:
        model = models.RAEDenoiser_models[args.model](
            img_size=args.img_size,
            num_classes=args.num_classes,
            noise_scale=args.noise_scale,
            config_path=getattr(args, "rae_config_path", None),
            collections_dir=getattr(args, "rae_collections_dir", None),
            rae_root=getattr(args, "rae_root", None),
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    # Broadcast weights from rank 0 before EMA init.
    if is_enabled():
        logger.info("[Model] Broadcasting weights from rank 0 ...")
        broadcast_module_params(model, src=0)
        logger.info("[Model] Broadcast done.")
    logger.info(f"====Model====\n{model}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} parameters: {n / 1e6:.2f}M ({n:,})")

    ema_values = args.ema_rates if args.ema_type == "const" else args.ema_halflife_kimg
    ema = EMAModel(model, ema_type=args.ema_type, values=ema_values, batch_size=args.global_bsz)
    logger.info(f"EMA: type={args.ema_type}, labels={ema.labels}")
    return model, ema


def create_tokenizer(args):
    """create, load weights, and optionally compile the tokenizer."""
    if args.tokenizer is None:
        logger.info("not using any tokenizer")
        return None
    logger.info(f"creating tokenizer: {args.tokenizer}")

    if args.tokenizer in models.VAE_models:
        tok = models.DiffusersAutoencoderKL(name=args.tokenizer)
    else:
        raise ValueError(f"unsupported tokenizer {args.tokenizer}")

    tok.cuda().eval().requires_grad_(False)
    if is_enabled():
        logger.info("[Tokenizer] Broadcasting weights from rank 0 ...")
        broadcast_module_params(tok, src=0)
        logger.info("[Tokenizer] Broadcast done.")
    return tok
