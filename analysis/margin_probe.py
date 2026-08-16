"""Margin-to-policy plausibility probe: task-demeaned OLS on archived selections."""
import glob, json, os, pickle, re, sys
import numpy as np
import torch
sys.path.insert(0, "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
os.chdir("/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data")
from model.policy import VEnsemble
import yaml

torch.set_num_threads(6)
cfg = yaml.safe_load(open("config.yaml"))
S = os.environ["SCRATCH"]
SNAP = json.load(open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/results_snapshot.json"))
LIMS = {"halfcheetah_velocity": 20, "walker2d_velocity": 20, "ant_velocity": 20,
        "swimmer_velocity": 20, "cargoal1_dsrl": 25, "cargoal2": 25,
        "pointgoal1_dsrl": 25, "pointgoal2": 25, "carrun_b": 10}
SPECS = []
for p in sorted(glob.glob(S + "/certified_h5/*_kept.json")):
    b = os.path.basename(p)[:-len("_kept.json")]
    m = re.match(r"(.+)_a40d([123])_seed(\d)$", b)
    if m:
        SPECS.append((m.group(1), p, int(m.group(3)), f"bc_a40d{m.group(2)}"))
    elif re.match(r"carrun_b_echocert_seed0$", b):
        SPECS.append(("carrun_b", p, 0, "bc_echo"))

score_cache = {}
rows = []
for task, kj, vs, bc_key in SPECS:
    if bc_key not in SNAP.get(task, {}):
        print(f"skip {task} {bc_key}: no BC arm", flush=True)
        continue
    key = (task, vs)
    if key not in score_cache:
        trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
        costs = np.array([float(np.sum(t["costs"])) for t in trajs])
        obs_dim = trajs[0]["observations"].shape[1]
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt",
                        map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, t in enumerate(trajs):
                o = torch.as_tensor(t["observations"], dtype=torch.float32)
                g[i] = torch.cat([ens(o[j:j+8192])
                                  for j in range(0, len(o), 8192)]).mean().item()
        score_cache[key] = ((g - g.mean()) / (g.std() + 1e-8), costs)
    z, costs = score_cache[key]
    kept = json.load(open(kj))["kept"]
    zk = z[kept]
    contam = float((costs[kept] > LIMS[task]).mean())
    ent = SNAP[task][bc_key]
    cost_norm = float(np.mean([e["C"] for e in ent.values()])) / LIMS[task]
    rows.append({"task": task, "sel": os.path.basename(kj), "contam": contam,
                 "mean_margin": float(zk.mean()),
                 "p10_margin": float(np.percentile(zk, 10)),
                 "cost_norm": cost_norm})
    print(f"{task} {bc_key}: contam={contam:.3f} mm={zk.mean():.2f} "
          f"p10={np.percentile(zk, 10):.2f} y={cost_norm:.2f}", flush=True)


def demean(rows, field):
    v = np.array([r[field] for r in rows])
    tasks = [r["task"] for r in rows]
    out = v.copy()
    for t in set(tasks):
        idx = [i for i, x in enumerate(tasks) if x == t]
        out[idx] -= v[idx].mean()
    return out


def r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0


y = demean(rows, "cost_norm")
Xc = demean(rows, "contam").reshape(-1, 1)
Xm = np.column_stack([Xc, demean(rows, "mean_margin"), demean(rows, "p10_margin")])
res = {"n_points": len(rows), "r2_contam": r2(Xc, y), "r2_full": r2(Xm, y)}
res["delta_r2"] = res["r2_full"] - res["r2_contam"]
res["rows"] = rows
json.dump(res, open("/home/omniverse/workspace/safevlmcpl/iclr2027/data/review_response/margin_probe.json", "w"), indent=1)
print(f"MARGIN PROBE: n={res['n_points']} R2(contam)={res['r2_contam']:.3f} "
      f"R2(full)={res['r2_full']:.3f} dR2={res['delta_r2']:.3f}", flush=True)
print("MARGIN PROBE DONE", flush=True)
