# Covariance conventions in EMA Fréchet loss

This note records the discussion around `N` versus `N - 1` in the EMA
Fréchet-loss workflow. There are three distinct places where covariance enters:

1. constructing fixed reference statistics;
2. initializing the generated running statistics; and
3. updating and scoring the generated EMA statistics.

The central distinction is between a **raw second moment** and a **covariance
estimate**. Raw moments must use `N`. A Bessel correction, when desired, is
applied only after centering.

## Notation

For features `x_1, ..., x_N`, define

```text
mu = (1 / N) sum_i x_i
M2 = (1 / N) sum_i x_i x_i^T
C_pop = M2 - mu mu^T
```

`M2` is an estimate of `E[x x^T]`, so its denominator is always `N`. It would
be incorrect to divide this raw moment by `N - 1`.

For an equally weighted finite sample, the usual sample covariance is

```text
C_sample = N / (N - 1) * (M2 - mu mu^T)
         = 1 / (N - 1) * sum_i (x_i - mu)(x_i - mu)^T.
```

Thus, when moments are stored in normalized form, using `N - 1` means
multiplying the centered raw-moment covariance by `N / (N - 1)`. It does not
mean dividing `M2 - mu mu^T` directly by `N - 1`.

## Current behavior

| Stage | Stored/computed quantity | Current convention |
|---|---|---|
| Reference creation | Exact centered covariance of a finite reference set | `N - 1` |
| Generated initialization | `mu` and `M2` from the initial generated set | Raw moments use `N`; implied covariance is `C_pop` |
| Generated EMA update | EMA of `mu` and `M2` | New raw moments use batch size `B`; covariance is `C_pop` |
| Exact evaluation | Exact centered covariance of a finite generated set | `N - 1` |

In `distmatch`, reference creation and exact evaluation call
`ExactMoments.mean_covariance()`, which uses `N - 1`. EMA initialization calls
`ExactMoments.mean_raw_second()`, which uses `N`. The training objective then
forms covariance as `M2_ema - mu_ema mu_ema^T` without a finite-weight
correction. This reproduces the original EMA FD-loss convention.

## Are the current choices valid?

There are two coherent interpretations.

### 1. Weighted empirical-distribution interpretation

Under this interpretation, the running state describes an exponentially
weighted empirical distribution. Its exact covariance is

```text
C_weighted = M2_ema - mu_ema mu_ema^T.
```

No Bessel correction is needed because this is the population covariance of
the weighted empirical distribution itself. The raw-moment EMA update is
algebraically correct and automatically includes covariance caused by changes
in historical generator means.

For complete convention consistency under this interpretation, reference
statistics would also use `N`, treating the finite reference dataset as the
target empirical population. That would differ slightly from standard FID
practice.

### 2. Underlying-population estimation interpretation

Standard FID normally treats real and generated features as samples from
underlying distributions and uses sample covariance. Under IID sampling,
`C_pop` is biased downward. For equal weights,

```text
E[C_pop] = (N - 1) / N * Cov[x].
```

For unequally weighted EMA samples, there is no single ordinary `N`. Let the
normalized individual sample weights be `w_i` and define

```text
q = sum_i w_i^2
N_eff = 1 / q.
```

The weighted Bessel correction is

```text
C_corrected = (M2_ema - mu_ema mu_ema^T) / (1 - q)
            = N_eff / (N_eff - 1) * C_weighted.
```

This is unbiased for a stationary IID population. It does not remove the
separate error caused by combining samples from changing historical
generators.

## How `q` evolves under EMA

Suppose the old normalized sample weights have squared sum `q_old`. An EMA
update assigns total weight `beta` to the old state and total weight
`1 - beta` equally across a new global batch of size `B`. Therefore,

```text
q_new = beta^2 * q_old + (1 - beta)^2 / B.
```

For an initialization set of size `N_init`,

```text
q_init = 1 / N_init.
```

The candidate `q_new` must be used for the candidate covariance evaluated by
the loss, and it should be committed alongside `mu_ema` and `M2_ema` in
`update_state()`.

This gives a continuous treatment of all beta values:

- `beta = 0`: `q_new = 1 / B`, so the correction becomes `B / (B - 1)`;
- `0 < beta < 1`: historical and current samples have unequal weights, so the
  correction depends on their effective sample size;
- large `beta`: many historical samples contribute, `N_eff` becomes large,
  and the correction approaches one.

The covariance rule is not arbitrarily changing with `beta`. The sample
weights, and therefore the appropriate weighted degrees-of-freedom correction,
change with `beta`.

## The `beta = 0` case

When `beta = 0`, initialization has zero influence on the loss. The generated
statistics come entirely from the current global batch, so standard
sample-covariance FD should use

```text
C = B / (B - 1) * (M2_batch - mu_batch mu_batch^T).
```

The size of this correction is `1 / (B - 1)` relative to the uncorrected
covariance:

| Global batch `B` | Relative correction |
|---:|---:|
| 1024 | 0.098% |
| 256 | 0.392% |
| 64 | 1.59% |
| 16 | 6.67% |

Consequently, the uncorrected implementation is not automatically harmless at
`beta = 0`; its accuracy depends directly on the logical global batch size.

## Magnitude under the intended EMA configuration

For the original configuration with `N_init = 50,000`, the initialization
correction is

```text
50,000 / 49,999 = 1.0000200004,
```

or approximately `0.002%`.

At steady state with `beta = 0.999` and global batch `B = 1024`,

```text
q = (1 - beta) / (B * (1 + beta)) ~= 4.89e-7,
```

so the correction is approximately `1.00000049`, or `0.000049%`. Sampling
noise, generator drift, and EMA lag are much larger sources of approximation
error in this regime.

## Practical conclusion

For compatibility with the original EMA FD-loss and its intended high-beta,
large-initialization configuration, the current uncorrected EMA covariance is a
reasonable simplification. The numerical error is negligible, and changing it
would break strict parity.

For a statistically consistent population-estimation mode:

1. continue storing all raw moments with denominator `N` or `B`;
2. keep reference creation and exact evaluation at `N - 1`;
3. track `q` as one additional checkpointed running scalar; and
4. divide the generated centered raw-moment covariance by `1 - q` when scoring
   FD.

This correction should be especially considered for `beta = 0`, small logical
batches, small initialization sets, or lower EMA decay. It fixes finite-sample
covariance bias, but it does not make EMA FD an exact FD objective for the
current generator: the running state still represents a temporally weighted
mixture of historical generators.
