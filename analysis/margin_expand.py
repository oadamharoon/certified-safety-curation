"""Margin expansion: control selections spanning the margin-contamination plane."""
import json, os, pickle, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
import yaml

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEV}", flush=True)
cfg = yaml.safe_load(open("config.yaml"))
SNAP = json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/results_snapshot.json"))
TASKS = [t for t in SNAP if "vfilt_return" in SNAP[t] and not t.endswith("_b")]
print(f"{len(TASKS)} main tasks", flush=True)


def score_all(task, trajs, seed):
    obs_dim = trajs[0]["observations"].shape[1]
    ens = VEnsemble(obs_dim, 256, K=3).to(DEV)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{seed}.pt",
                    map_location=DEV, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    ens.eval()
    g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32, device=DEV)
            g[i] = torch.cat([ens(o[j:j+16384])
                              for j in range(0, len(o), 16384)]).mean().item()
    return (g - g.mean()) / (g.std() + 1e-8)


rows = list(json.load(open(
    "/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/margin_probe.json"))["rows"])
for task in TASKS:
    lim = 20 if "velocity" in task else 25
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    R = np.array([float(np.sum(t["rewards"])) for t in trajs])
    safe_mask = costs <= lim
    frac = float(safe_mask.mean())
    nsel = max(1, int(round(frac * len(trajs))))
    z0 = score_all(task, trajs, 0)
    order_R = np.argsort(R)[::-1]

    def add(sel, cfgname, seed_key=None, z=None):
        ent = SNAP[task].get(cfgname, {})
        if not ent:
            print(f"miss {task} {cfgname}", flush=True)
            return
        if seed_key is not None:
            if seed_key not in ent:
                print(f"miss {task} {cfgname} seed {seed_key}", flush=True)
                return
            y = ent[seed_key]["C"] / lim
        else:
            y = float(np.mean([e["C"] for e in ent.values()])) / lim
        zz = z0 if z is None else z
        rows.append({"task": task, "sel": f"{cfgname}:{seed_key}",
                     "contam": float((costs[sel] > lim).mean()),
                     "mean_margin": float(zz[sel].mean()),
                     "p10_margin": float(np.percentile(zz[sel], 10)),
                     "cost_norm": float(y)})

    add(np.arange(len(trajs)), "bc_all")
    add(np.where(safe_mask)[0], "bcsafe")
    add(order_R[:nsel], "vfilt_return")
    add(order_R[::-1][:nsel], "vfilt_retbot")
    for s in range(5):
        try:
            zs = z0 if s == 0 else score_all(task, trajs, s)
        except FileNotFoundError:
            print(f"miss {task} ensemble seed {s}", flush=True)
            continue
        top = np.argsort(zs)[::-1]
        add(top[:nsel], "vfilt_matchgt", str(s), zs)
        add(top[:max(1, int(round(0.25 * len(trajs))))], "vfilt_q25", str(s), zs)
    print(f"{task}: rows now {len(rows)}", flush=True)


def demean(rows, field):
    v = np.array([r[field] for r in rows])
    tasks = [r["task"] for r in rows]
    out = v.copy()
    for t in set(tasks):
        idx = [i for i, x in enumerate(tasks) if x == t]
        out[idx] -= v[idx].mean()
    return out


def fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def r2_of(X, y, beta=None):
    X1 = np.column_stack([np.ones(len(y)), X])
    if beta is None:
        beta = fit(X, y)
    resid = y - X1 @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / ss) if ss > 0 else 0.0


y = demean(rows, "cost_norm")
Xc = demean(rows, "contam").reshape(-1, 1)
Xm = np.column_stack([Xc, demean(rows, "mean_margin"), demean(rows, "p10_margin")])
res = {"n_points": len(rows), "r2_contam": r2_of(Xc, y), "r2_full": r2_of(Xm, y)}
res["delta_r2"] = res["r2_full"] - res["r2_contam"]
# leave-one-task-out
tasks_arr = np.array([r["task"] for r in rows])
sse_c = sse_m = sst = 0.0
for t in set(tasks_arr):
    tr, te = tasks_arr != t, tasks_arr == t
    bc = fit(Xc[tr], y[tr]); bm = fit(Xm[tr], y[tr])
    Xc1 = np.column_stack([np.ones(te.sum()), Xc[te]])
    Xm1 = np.column_stack([np.ones(te.sum()), Xm[te]])
    sse_c += ((y[te] - Xc1 @ bc) ** 2).sum()
    sse_m += ((y[te] - Xm1 @ bm) ** 2).sum()
    sst += ((y[te] - y[~te].mean()) ** 2).sum()
res["loto_r2_contam"] = float(1 - sse_c / sst)
res["loto_r2_full"] = float(1 - sse_m / sst)
res["rows"] = rows
json.dump(res, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/margin_expand.json", "w"), indent=1)
print(f"MARGIN EXPAND: n={res['n_points']} R2c={res['r2_contam']:.3f} "
      f"R2full={res['r2_full']:.3f} dR2={res['delta_r2']:.3f} "
      f"LOTO {res['loto_r2_contam']:.3f}->{res['loto_r2_full']:.3f}", flush=True)
print("MARGIN EXPAND DONE", flush=True)
