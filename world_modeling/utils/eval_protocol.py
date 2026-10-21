"""Historical rollout identity and metric summary conventions."""
from pathlib import Path
N_FVD_FRAMES = 16

def _build_run_root(cfg, ckpt_tag, cfg_scale, mode):
    name = f"{ckpt_tag}_cfg{cfg_scale}_{mode}"
    results_dir = cfg.get("results_dir", cfg.output_dir)
    return Path(results_dir) / "eval_on_validation_multigpu" / name


def _sampling_mode(cfg, chunk_gen_frames, flow_steps=None, flow_solver=None):
    """Resolve sampling controls and a collision-free results-directory tag."""
    is_flow = str(getattr(cfg.model, "objective", "drift")) == "flow_matching"
    if not is_flow:
        if flow_steps is not None or flow_solver is not None:
            raise ValueError("--flow-steps/--flow-solver require a flow_matching config")
        return f"chunkAuto{chunk_gen_frames}", None, None

    steps = int(cfg.flow.sampling_steps if flow_steps is None else flow_steps)
    solver = str(cfg.flow.solver if flow_solver is None else flow_solver).lower()
    if steps < 1:
        raise ValueError(f"flow_steps must be positive, got {steps}")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"flow_solver must be euler or heun, got {solver!r}")
    return f"flow-{solver}-{steps}step_chunkAuto{chunk_gen_frames}", steps, solver


def _config_resolved(config_path) -> str:
    """Stable config identity for resume checks."""
    return str(Path(config_path).expanduser().resolve())


def _build_resume_metadata(config_path, rank, world_size, actual_step, *, cfg_scale,
                           n_history, fps, chunk_size, max_videos_per_shard,
                           chunk_gen_frames, flow_steps=None, flow_solver=None):
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "actual_step": int(actual_step),
        "config_resolved": _config_resolved(config_path),
        "cfg_scale": float(cfg_scale),
        "n_history": int(n_history),
        "fps": int(fps),
        "chunk_size": int(chunk_size),
        "max_videos_per_shard": max_videos_per_shard,
        "chunk_gen_frames": int(chunk_gen_frames),
        "flow_steps": flow_steps,
        "flow_solver": flow_solver,
    }


def _record_totals(records: list):
    total_gen_time = sum(float(r.get("gen_time_s", 0.0)) for r in records)
    total_gen_frames = sum(int(r.get("n_gen", 0)) for r in records)
    return total_gen_time, total_gen_frames


def _running_summary(records: list) -> str:
    """Running averages over all records completed in this shard so far."""
    n = len(records)
    if n == 0:
        return "no clips yet"
    total_gen_time, total_gen_frames = _record_totals(records)
    tpf = total_gen_time / total_gen_frames if total_gen_frames > 0 else float('nan')
    n_fvd = sum(1 for r in records if int(r.get("fvd_kept_frames", 0)) >= N_FVD_FRAMES)

    def _mean(key):
        return sum(float(r[key]) for r in records) / n

    return (f"n={n} | SSIM={_mean('ssim'):.4f} "
            f"PSNR={_mean('psnr'):.4f} LPIPS={_mean('lpips'):.4f} | "
            f"time/frame={tpf:.4f}s | fvd-eligible={n_fvd}/{n}")
