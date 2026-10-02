"""Post-evaluation publication, isolated from optimizer and training streams."""
from pathlib import Path


def matches_compute_audit(args):
    """Only reuse the B count for the measured architecture and training lineage."""
    from src.eval.dashboard_data import ROOT
    expected = dict(model="OST_B", dashboard_compute_model="OST_B", img_size=256,
        patch_size=16, class_tokens=8, time_tokens=0, guidance_tokens=0,
        interval_tokens=0, aux_head_depth=8, token_channels=3, tokenizer=None,
        train_last_n_blocks=None, use_lora=False, grad_checkpointing=False,
        ot_feature_stack="convnext_full_mae", discrepancy_kind="featurewise_ot",
        global_bsz=1024, world_size=8, grad_accum=1, ot_num_classes=16,
        ot_samples_per_class=64, ot_num_real_refs=128, ot_encoder_microbatch=8,
        ot_encoder_checkpointing=False, ot_compile_encoders=False,
        ot_feature_gradient_replay=True, ot_baseline_weight=0.0, xm_enabled=False,
        mmd_repulsion_queue_weight=0.0, dtype="bf16", compile=True)
    if any(getattr(args, key, object()) != value for key, value in expected.items()):
        return False
    source = ROOT / "fdloss_work_dirs/OST/past/OST_B_aux_unif_t08_20260629_140029/01_ost/checkpoints/step_0100080.pth"
    return bool(args.load_from) and Path(args.load_from).resolve() == source.resolve()


def publish_completed_run(args, model, ema):
    import torch.distributed as distributed
    from scripts.build_dashboards import register_run, refresh
    from src.eval.dashboard_data import ROOT, write_json
    from src.eval.sample_dashboard import export_ost_panel
    from src.utils.distributed_util import get_global_rank, is_enabled

    if not getattr(args, "dashboard_enabled", False):
        return
    if args.current_step != args.total_steps or args.skip_final_eval:
        raise ValueError("Dashboard completion requires the terminal checkpoint and evaluation")
    error = None
    if get_global_rank() == 0:
        status = Path(args.log_dir) / "dashboard_status.json"
        try:
            output = ROOT / "figures/samples" / args.exp_name
            checkpoint = Path(args.ckpt_dir) / f"step_{args.current_step-1:07d}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Terminal checkpoint is missing: {checkpoint}")
            export_ost_panel(args, model, ema, output, step=args.current_step,
                checkpoint=checkpoint, label=args.model.replace("_", "-"))
            # The B audit is valid only for its measured recipe, never for L scaling.
            matches_audit = matches_compute_audit(args)
            register_run(args.log_dir, args.current_step,
                pretrain_eflops=args.dashboard_pretrain_eflops if matches_audit else None,
                step_tflops=args.dashboard_step_tflops if matches_audit else None,
                compute_source=args.dashboard_compute_source if matches_audit else None)
            refresh()
            write_json(status, dict(status="completed", step=args.current_step,
                benchmark=str(ROOT / "figures/benchmarks/index.html"),
                samples=str(output / "index.html"),
                run=str(ROOT / "figures/runs" / args.exp_name / "index.html")))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            write_json(status, dict(status="failed", step=args.current_step, error=error))
    if is_enabled():
        payload = [error]
        distributed.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error:
        raise RuntimeError(f"Training and evaluation finished; dashboard publication failed: {error}")
