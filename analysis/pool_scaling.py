"""Pool-size scaling: sequential e-process labels vs minimal fixed n."""
import json
import numpy as np
from scipy.stats import hypergeom
from scipy.special import expit

ALPHA, DELTA, REPS = 0.25, 0.1, 500
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
THRESH = 1.0 / DELTA
rng = np.random.default_rng(77)


def make_pool(N, safe_mass):
    g = rng.standard_normal(N)
    b = 2.2
    lo, hi = -12.0, 12.0
    for _ in range(60):
        a = (lo + hi) / 2
        if expit(a - b * g).mean() > 1 - safe_mass:
            hi = a
        else:
            lo = a
    unsafe = (rng.random(N) < expit(a - b * g)).astype(float)
    return g, unsafe


def fixed_power(g, unsafe, n, reps=REPS):
    N = len(g)
    taus = np.quantile(g, QS)
    Ns = np.array([(g >= t).sum() for t in taus])
    Kstar = np.floor(ALPHA * Ns) + 1
    nc = 0
    for _ in range(reps):
        cal = rng.choice(N, size=n, replace=False)
        cs, cu = g[cal], unsafe[cal]
        ok = False
        for j, t in enumerate(taus):
            sel = cs >= t
            m, k = int(sel.sum()), int(cu[sel].sum())
            p = 1.0 if (m == 0 or Kstar[j] > Ns[j]) else \
                float(hypergeom.cdf(k, Ns[j], int(Kstar[j]), m))
            if m > 0 and p <= DELTA:
                ok = True
            else:
                break
        nc += ok
    return nc / reps


def eprocess(g, unsafe, cap, reps=REPS):
    N, J = len(g), len(QS)
    taus = np.quantile(g, QS)
    Ns = np.array([(g >= t).sum() for t in taus]).astype(float)
    Kstar = np.floor(ALPHA * Ns) + 1
    perms = np.stack([rng.permutation(N)[:cap] for _ in range(reps)])
    gsel, usel = g[perms], unsafe[perms]
    W = np.ones((reps, J)); ic = np.zeros((reps, J)); sc = np.zeros((reps, J))
    crossed = np.zeros((reps, J), dtype=bool)
    t_cert = np.full(reps, -1); active = np.ones(reps, dtype=bool)
    for t in range(cap):
        gt, ut = gsel[:, t], usel[:, t]
        for j in range(J):
            inS = (gt >= taus[j]) & active
            if not inS.any():
                continue
            rem = Ns[j] - ic[:, j]
            mu = np.clip((Kstar[j] - sc[:, j]) / np.maximum(rem, 1), 1e-6, 1 - 1e-6)
            phat = (sc[:, j] + 0.5) / (ic[:, j] + 1.0)
            lam = np.clip((mu - phat) / (mu * (1 - mu) + 1e-4), 0.0, 0.5 / (1 - mu))
            W[:, j] = np.where(inS, W[:, j] * (1 + lam * (mu - ut)), W[:, j])
            ic[:, j] += inS
            sc[:, j] += inS * ut
            crossed[:, j] |= W[:, j] >= THRESH
        newly = active & crossed[:, 0]
        t_cert[newly] = t + 1
        active[newly] = False
        if not active.any():
            break
    cert = t_cert > 0
    return (float(cert.mean()),
            float(np.median(t_cert[cert])) if cert.any() else None)


out = {}
for safe_mass in (0.3, 0.5):
    for N in (1000, 3000, 10000, 30000, 100000):
        g, unsafe = make_pool(N, safe_mass)
        lo, hi = 25, min(N // 2, 4000)
        if fixed_power(g, unsafe, hi) < 0.9:
            nmin = None
        else:
            while hi - lo > 10:
                mid = (lo + hi) // 2
                if fixed_power(g, unsafe, mid, reps=250) >= 0.9:
                    hi = mid
                else:
                    lo = mid
            nmin = hi
        if nmin is None:
            out[f"{safe_mass}_{N}"] = {"n_fixed_min": None}
            print(f"mass={safe_mass} N={N}: fixed-n never reaches 0.9 power", flush=True)
            continue
        rate, med = eprocess(g, unsafe, cap=min(4 * nmin, N))
        ratio = (med / nmin) if med else None
        out[f"{safe_mass}_{N}"] = {"n_fixed_min": nmin, "seq_cert_rate": rate,
                                   "seq_median": med, "ratio": ratio}
        print(f"mass={safe_mass} N={N}: n_fixed={nmin} seq_rate={rate:.2f} "
              f"seq_median={med} ratio={ratio}", flush=True)
json.dump(out, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/pool_scaling.json", "w"), indent=1)
print("POOL SCALING DONE", flush=True)
