import logging
from collections import defaultdict
from pathlib import Path

import src.utils.muon_patch  # noqa: F401  — fixes Muon distributed bugs
from muon import MuonWithAuxAdam

import torch

logger = logging.getLogger("FD_loss")


def create_optimizer(args, model, print_trainable_params=False):
    logger.info("creating optimizer")
    eff_bs = getattr(args, "global_bsz", args.batch_size * args.world_size)
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "muon":
        return create_muon_optimizer(args, model, eff_bs, print_trainable_params)
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return create_adamw_optimizer(args, model, eff_bs, print_trainable_params)


def create_adamw_optimizer(args, model, eff_bs, print_trainable_params=False):
    exclude = lambda n, p: (
        p.ndim < 2 or any(k in n for k in
        ("ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"))
    )
    named = list(model.named_parameters())
    nodecay = [p for n, p in named if exclude(n, p) and p.requires_grad]
    decay = [p for n, p in named if not exclude(n, p) and p.requires_grad]

    if args.lr is None:
        args.lr = args.blr * eff_bs / 256

    logger.info(f"base lr: {args.lr * 256 / eff_bs:.6e}, actual lr: {args.lr:.6e}, lr_sched: {args.lr_sched}")
    logger.info(f"eff batch size: {eff_bs}, gpus: {args.world_size}")
    logger.info(f"weight_decay={args.weight_decay} on {len(decay)} tensors, no_decay on {len(nodecay)}")

    opt = torch.optim.AdamW(
        [{"params": nodecay, "weight_decay": 0.0},
         {"params": decay, "weight_decay": args.weight_decay}],
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    logger.info(f"optimizer = {opt}")

    if print_trainable_params:
        decay_np = [(n, p) for n, p in named if not exclude(n, p) and p.requires_grad]
        nodecay_np = [(n, p) for n, p in named if exclude(n, p) and p.requires_grad]
        for n, _ in decay_np:  logger.info(f"\t\\[adamw+wd={args.weight_decay}] {n}")
        for n, _ in nodecay_np: logger.info(f"\t\\[adamw] {n}")
        save_param_groups(Path(args.log_dir) / "params_group.txt", model, [
            {"label": f"adamw (decay={args.weight_decay})",
             "names": [n for n, _ in decay_np], "params": [p for _, p in decay_np],
             "lr": args.lr, "wd": args.weight_decay},
            {"label": "adamw (no decay)",
             "names": [n for n, _ in nodecay_np], "params": [p for _, p in nodecay_np],
             "lr": args.lr, "wd": 0.0},
        ])
    return opt


def create_muon_optimizer(args, model, eff_bs, print_trainable_params=False):
    muon_params, adamw_params, muon_names, adamw_names = get_muon_param_groups(model)

    counts = {label: sum(p.numel() for p in ps) for label, ps in
              [("muon", muon_params), ("adamw", adamw_params)]}
    total = sum(counts.values())

    logger.info(f"eff batch size: {eff_bs}, gpus: {args.world_size}")
    logger.info("=== pMF-style muon optimizer ===")
    for label, c in counts.items():
        pct = 100 * c / total if total else 0.0
        logger.info(f"  {label}: {c:,} params ({pct:.1f}%)")
    logger.info(
        f"lr={args.lr}, muon momentum={args.muon_momentum}, "
        f"muon wd={args.muon_weight_decay}, adamw betas=({args.beta1}, {args.beta2})"
    )

    if print_trainable_params:
        save_param_groups(Path(args.log_dir) / "params_group.txt", model, [
            {"label": "muon", "names": muon_names, "params": muon_params,
             "lr": args.lr, "wd": args.muon_weight_decay},
            {"label": "adamw", "names": adamw_names,
             "params": adamw_params, "lr": args.lr, "wd": 0.0},
        ])

    adamw_kw = dict(lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)
    groups = []
    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=args.lr,
                momentum=args.muon_momentum,
                weight_decay=args.muon_weight_decay,
            )
        )
    if adamw_params:
        groups.append(dict(params=adamw_params, use_muon=False, weight_decay=0.0, **adamw_kw))
    opt = MuonWithAuxAdam(groups)
    logger.info(f"optimizer = {opt}")
    return opt


def save_param_groups(path: Path, model, groups: list[dict]):
    """Write a pretty parameter-group summary + detailed table to *path*.

    Args:
        model: the nn.Module (used to detect frozen / non-learnable params).
        groups: list of dicts, each with keys
            "label" (str), "names" (list[str]), "params" (list[Tensor]),
            "lr" (float), "wd" (float).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # ---- collect frozen (non-learnable) params ----
    trainable_names = {n for g in groups for n in g["names"]}
    frozen = [(n, p) for n, p in model.named_parameters() if n not in trainable_names]

    # ---- per-group counts ----
    grp_counts = []  # (label, count, lr, wd)
    total_train = 0
    for g in groups:
        cnt = sum(p.numel() for p in g["params"])
        grp_counts.append((g["label"], cnt, g["lr"], g["wd"]))
        total_train += cnt
    total_frozen = sum(p.numel() for _, p in frozen)
    total_all = total_train + total_frozen

    # row: (group, name, lr, wd, shape_str, count)
    rows = []
    for g in groups:
        for name, p in zip(g["names"], g["params"]):
            rows.append((g["label"], name, g["lr"], g["wd"],
                         str(tuple(p.shape)), p.numel()))

    # frozen row: (name, shape_str, count)
    frozen_rows = []
    for n, p in frozen:
        frozen_rows.append((n, str(tuple(p.shape)), p.numel()))

    # ---- build hierarchical module summary ----
    all_params = [(name, p.numel()) for g in groups for name, p in zip(g["names"], g["params"])]
    all_params += [(n, p.numel()) for n, p in frozen]
    leaf_names = {name for name, _ in all_params}
    prefix_counts = defaultdict(int)
    for name, cnt in all_params:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix_counts[".".join(parts[:i])] += cnt
    # remove leaf param names -- they appear in the detailed table
    for name in leaf_names:
        prefix_counts.pop(name, None)

    # ---- compute column widths for the detailed table ----
    def _col_w(idx, header, items=rows):
        return max(max((len(r[idx]) for r in items), default=len(header)), len(header))

    w_grp   = _col_w(0, "Group")
    w_name  = _col_w(1, "Parameter Name")
    w_shape = _col_w(4, "Shape")

    with open(path, "w") as f:
        # ============== optimizer group summary ==============
        f.write("Parameter groups summary:\n")
        sh = f"{'Group':<25s}  {'Params':>14s}  {'% Total':>8s}  {'lr':>10s}  {'wd':>10s}\n"
        ss = f"{'-'*25}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*10}\n"
        f.write(sh)
        f.write(ss)
        for label, cnt, lr, wd in grp_counts:
            pct = 100 * cnt / total_train if total_train else 0
            f.write(f"{label:<25s}  {cnt:>14,}  {pct:>7.2f}%  {lr:>10g}  {wd:>10g}\n")
        f.write(ss)
        f.write(f"{'Total (trainable)':<25s}  {total_train:>14,}  {'100.00%':>8s}\n")
        if frozen:
            pct_f = 100 * total_frozen / total_all if total_all else 0
            f.write(f"{'Frozen (non-learnable)':<25s}  {total_frozen:>14,}  {pct_f:>7.2f}%\n")
        f.write(f"{'Total (all)':<25s}  {total_all:>14,}\n")
        f.write("\n")

        # ============== hierarchical module summary ==============
        if prefix_counts:
            f.write("Module parameter summary:\n")
            sorted_prefixes = sorted(prefix_counts.keys())
            w_mod = max(len(p) + p.count(".") for p in sorted_prefixes)
            w_mod = max(w_mod, len("Module"))
            mh = f"{'Module':<{w_mod}s}  {'Params':>14s}  {'% Total':>8s}\n"
            ms = f"{'-'*w_mod}  {'-'*14}  {'-'*8}\n"
            f.write(mh)
            f.write(ms)
            for prefix in sorted_prefixes:
                depth = prefix.count(".")
                indent = " " * depth
                cnt = prefix_counts[prefix]
                pct = 100 * cnt / total_all if total_all else 0
                display = f"{indent}{prefix}"
                f.write(f"{display:<{w_mod}s}  {cnt:>14,}  {pct:>7.2f}%\n")
            f.write("\n")

        # ============== detailed table ==============
        f.write("Detailed parameter breakdown:\n")
        hdr = (f"{'Group':<{w_grp}s}  {'Parameter Name':<{w_name}s}  "
               f"{'Shape':>{w_shape}s}  {'Count':>12s}  {'% Total':>8s}  "
               f"{'lr':>10s}  {'wd':>10s}\n")
        sep = (f"{'-'*w_grp}  {'-'*w_name}  {'-'*w_shape}  {'-'*12}  "
               f"{'-'*10}  {'-'*10}\n")
        f.write(hdr)
        f.write(sep)
        for grp, name, lr, wd, shape, cnt in rows:
            pct = 100 * cnt / total_train if total_train else 0
            f.write(f"{grp:<{w_grp}s}  {name:<{w_name}s}  "
                    f"{shape:>{w_shape}s}  {cnt:>12,}  {pct:>7.2f}%  "
                    f"{lr:>10g}  {wd:>10g}\n")

        # ============== frozen params ==============
        if frozen_rows:
            f.write("\n")
            f.write("Non-learnable parameters:\n")
            def _fw(idx, hdr):
                return max(max((len(r[idx]) for r in frozen_rows), default=len(hdr)), len(hdr))
            w_fn = _fw(0, "Parameter Name")
            w_fs = _fw(1, "Shape")
            fh = (f"{'Parameter Name':<{w_fn}s}  {'Shape':>{w_fs}s}  "
                  f"{'Count':>12s}  {'lr':>10s}  {'wd':>10s}\n")
            fs = f"{'-'*w_fn}  {'-'*w_fs}  {'-'*12}  {'-'*10}  {'-'*10}\n"
            f.write(fh)
            f.write(fs)
            for row in frozen_rows:
                name, shape, cnt = row[0], row[1], row[2]
                f.write(f"{name:<{w_fn}s}  {shape:>{w_fs}s}  {cnt:>12,}  "
                        f"{'N/A':>10s}  {'N/A':>10s}\n")

    logger.info(f"param groups saved to {path}")


def get_muon_param_groups(model):
    """Match pMF/Optax Muon grouping for the PyTorch port.

    In upstream pMF, learnable tokens and label embeddings are rank-3 leaves, so
    optax.contrib.muon sends them to AdamW.  The PyTorch port stores several of
    those tensors as rank-2 parameters, so keep them on AdamW by name.
    """
    muon_params, adamw_params = [], []
    muon_names, adamw_names = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_pmf_adam_leaf = any(k in name for k in ("pos_embed", "token", "in_context", "embedding"))
        if p.ndim == 2 and not is_pmf_adam_leaf:
            muon_params.append(p); muon_names.append(name)
        else:
            adamw_params.append(p); adamw_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names
