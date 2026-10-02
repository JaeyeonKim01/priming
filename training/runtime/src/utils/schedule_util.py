import math

def adjust_learning_rate(optimizer, step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        lr = args.lr * step / args.warmup_steps
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            progress = (step - args.warmup_steps) / (args.total_steps - args.warmup_steps)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
