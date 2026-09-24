"""F2 (V2_REMEDIATION): certified selections of the six regenerated tasks from the ensembles at the
stated protocol, in one place and under one rule, plus their DSRL-format hdf5 subsets for CDT.

Per task:
  deployed alpha=.25 selection : kept_calfilt_csf_seed{vs}.json of the LOWEST certifying seed vs
                                 (the deployed-draw rule of the paper) -> <task>_cert_seed{vs}
  three distinct alpha=.25 sel.: the three most probable distinct certified thresholds over
                                 500 resampled n=200 draws at V seed vs (build_selections.py's
                                 rule) -> <task>_a25new_q{qq}
  deployed alpha=.40 selection : kept_calfilt_a40_seed{vs40}.json, lowest certifying seed
                                 -> <task>_a40_seed{vs40}
  three distinct alpha=.40 sel.: same rule at alpha=.40, V seed vs40 -> <task>_a40new_q{qq}
A task with no certifying csf seed gets no alpha=.25 entries; one with no certifying a40 seed
gets no alpha=.40 entries. Every subset is written on the raw-DSRL span route of build_cert_h5.py
(real terminals). Output: runs/selections/<name>_kept.json, <name>.hdf5, runs/selections/v2_summary.json.
Idempotent: existing hdf5 are not rebuilt.
"""

# --- paths ------------------------------------------------------------------
# The research tree addressed itself by absolute path; these roots replace it. Set
# CSC_WORKSPACE (or the individual roots) to point at your own trees. See the README.
import os as _os


def _csc_root(_p):
    """The repository root, found by the .csc-root marker rather than by depth."""
    _d = _os.path.dirname(_os.path.abspath(_p))
    while True:
        if _os.path.exists(_os.path.join(_d, ".csc-root")):
            return _d
        _up = _os.path.dirname(_d)
        if _up == _d:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(_p)))
        _d = _up


# __file__ is undefined when a script's source is exec'd in a fresh namespace, which the
# audit does to reuse the table builder's tables; fall back to the working directory, which
# the .csc-root walk resolves from anywhere inside the repository.
_self = globals().get("__file__") or _os.path.join(_os.getcwd(), "_")
CSC_REPO = _os.environ.get("CSC_REPO", _csc_root(_self))
_WS = _os.environ.get("CSC_WORKSPACE", _os.path.dirname(CSC_REPO))
CSC_WORK = _os.environ.get("CSC_WORK", _os.path.join(_WS, "vlm-with-cpl", "new_data"))
_runs = _os.path.join(_WS, "runs")
CSC_RUNS = _os.environ.get("CSC_RUNS", _runs if _os.path.isdir(_runs) else _os.path.join(CSC_REPO, "runs"))
CSC_OSRL = _os.environ.get("CSC_OSRL", _os.path.join(_WS, "osrl"))
CSC_PAPER = _os.environ.get("CSC_PAPER", _os.path.join(CSC_REPO, "paper"))
CSC_PAPER_DATA = _os.path.join(CSC_PAPER, "data")
# the run configs are carried by the repository, so they resolve without a working tree
CSC_CONFIG = _os.environ.get("CSC_CONFIG", _os.path.join(CSC_REPO, "configs"))
# -----------------------------------------------------------------------------

import json, os, pickle, sys
import numpy as np, torch, yaml, h5py
from scipy.stats import hypergeom
D = CSC_WORK; OUT = CSC_RUNS + "/selections"
sys.path.insert(0, D); os.chdir(D); os.makedirs(OUT, exist_ok=True)
from model.policy import VEnsemble
import gymnasium as gym
try:
    import dsrl
    if hasattr(dsrl, "register_envs"): dsrl.register_envs()
except Exception:
    pass
cfg = yaml.safe_load(open("config.yaml"))
TASKS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["halfcheetah_velocity", "cargoal1_dsrl", "walker2d_velocity", "ant_velocity", "hopper_velocity", "swimmer_velocity"]
QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]; DELTA, NCAL, REPS = 0.1, 200, 500
dev = "cuda" if torch.cuda.is_available() else "cpu"


def scores(task, trajs, vs):
    ens = VEnsemble(trajs[0]["observations"].shape[1], 256, K=3).to(dev)
    ck = torch.load(f"outputs/{task}/v_ensemble_pess_seed{vs}.pt", map_location=dev, weights_only=False)
    ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ens.eval(); g = np.zeros(len(trajs))
    with torch.no_grad():
        for i, t in enumerate(trajs):
            o = torch.as_tensor(t["observations"], dtype=torch.float32, device=dev)
            g[i] = torch.cat([ens(o[j:j + 8192]) for j in range(0, len(o), 8192)]).mean().item()
    return g


def distinct(g, unsafe, alpha, vs):
    taus = [float(np.quantile(g, q)) for q in QS]; Ns = [int((g >= t).sum()) for t in taus]
    rng = np.random.default_rng(500 + vs); picks = {}
    for _ in range(REPS):
        cal = rng.choice(len(g), NCAL, replace=False); cs, cu = g[cal], unsafe[cal]; chosen = None
        for j, t in enumerate(taus):
            sel = cs >= t; m, k = int(sel.sum()), int(cu[sel].sum()); ks = int(alpha * Ns[j]) + 1
            p = float(hypergeom.cdf(k, Ns[j], ks, m)) if (m > 0 and ks <= Ns[j]) else 1.0
            if m > 0 and p <= DELTA: chosen = j
            else: break
        if chosen is not None: picks[chosen] = picks.get(chosen, 0) + 1
    top = sorted(picks.items(), key=lambda kv: -kv[1])[:3]
    return sum(picks.values()) / REPS, [(QS[j], cnt / REPS, np.where(g >= taus[j])[0]) for j, cnt in top]


def lowest_certifying(task, arm, seeds):
    for s in seeds:
        p = f"outputs/{task}/calfilt_meta_{arm}_seed{s}.json"
        if os.path.exists(p) and json.load(open(p)).get("certified"): return s
    return None


def write_h5(task, tc, trajs, kept_idx, name, d, spans, term, tout):
    outp = f"{OUT}/{name}.hdf5"
    json.dump({"kept": [int(i) for i in kept_idx]}, open(f"{OUT}/{name}_kept.json", "w"))
    if os.path.exists(outp): return
    keep = np.zeros(len(trajs), bool); keep[np.asarray(kept_idx, int)] = True
    idx = np.concatenate([np.arange(s, e + 1) for (s, e), k in zip(spans, keep) if k])
    with h5py.File(outp, "w") as f:
        for key in ("observations", "actions", "rewards", "costs"): f.create_dataset(key, data=np.asarray(d[key])[idx])
        if "next_observations" in d: f.create_dataset("next_observations", data=np.asarray(d["next_observations"])[idx])
        else:
            obs = np.asarray(d["observations"]); nxt = obs.copy(); nxt[:-1] = obs[1:]
            for s_, e_ in spans: nxt[e_] = obs[e_]
            f.create_dataset("next_observations", data=nxt[idx])
        f.create_dataset("terminals", data=term[idx]); f.create_dataset("timeouts", data=tout[idx])


summary = json.load(open(f"{OUT}/v2_summary.json")) if os.path.exists(f"{OUT}/v2_summary.json") else {}
for task in TASKS:
    tc = cfg["tasks"][task]; lim = tc.get("cost_limit", cfg["cost_limit"]); trajs = pickle.load(open(tc["data_pickle"], "rb"))
    cost = np.array([float(np.sum(t["costs"])) for t in trajs]); unsafe = (cost > lim).astype(float)
    name = tc.get("offline_env_name") or tc["env_name"].replace("Safety", "Offline", 1)
    env = gym.make(name); d = env.get_dataset(); env.close(); term = np.asarray(d["terminals"], bool)
    if "timeouts" in d: tout = np.asarray(d["timeouts"], bool); end = np.logical_or(term, tout)
    else:
        end = term.copy(); end[999::1000] = True; tout = np.zeros_like(term); tout[999::1000] = True
    if not end[-1]: end[-1] = True
    ends = np.where(end)[0]; starts = np.concatenate([[0], ends[:-1] + 1]); spans = [(s, e) for s, e in zip(starts, ends) if e + 1 - s >= 2]
    assert len(spans) == len(trajs), f"{task}: span/pickle mismatch {len(spans)} vs {len(trajs)}"
    rec = {"limit": lim, "n": len(trajs)}
    for alpha, arm, seeds, tag in ((0.25, "calfilt_csf", range(5), "a25new"), (0.40, "calfilt_a40", range(3), "a40new")):
        vs = lowest_certifying(task, arm, seeds)
        if vs is None:
            print(f"{task} alpha {alpha}: no certifying seed", flush=True); rec[f"a{int(alpha*100)}"] = None; continue
        # 04q writes no membership file; the deployed selection is scores >= tau of the run's meta
        # (min 50), recomputed from the ensemble the run used. The old kept_calfilt_*.json files of
        # these tasks belong to the archived cohort and are not read.
        meta = json.load(open(f"outputs/{task}/calfilt_meta_{arm}_seed{vs}.json")); g = scores(task, trajs, vs)
        kept = np.where(g >= meta["tau"])[0]
        if len(kept) < 50: kept = np.argsort(g)[::-1][:50]
        assert len(kept) == meta["n_kept"] and abs(unsafe[kept].mean() - meta["kept_unsafe_rate"]) < 1e-6, (task, arm, vs, len(kept), meta["n_kept"])
        dep = f"{task}_cert_seed{vs}" if alpha == 0.25 else f"{task}_a40_seed{vs}"
        write_h5(task, tc, trajs, kept, dep, d, spans, term, tout)
        rate, sels = distinct(g, unsafe, alpha, vs)
        entry = {"v_seed": vs, "deployed": {"name": dep, "n": len(kept), "contamination": float(unsafe[kept].mean())}, "cert_rate_500": rate, "distinct": []}
        for q, prob, idx in sels:
            nm = f"{task}_{tag}_q{int(q*100)}"; write_h5(task, tc, trajs, idx, nm, d, spans, term, tout)
            entry["distinct"].append({"name": nm, "q": q, "prob": prob, "n": int(len(idx)), "contamination": float(unsafe[idx].mean())})
        entry["distinct"].sort(key=lambda s: s["contamination"]); rec[f"a{int(alpha*100)}"] = entry
        print(f"{task} alpha {alpha}: V seed {vs}, deployed n={len(kept)} ahat={unsafe[kept].mean():.3f}, resampled rate {rate:.2f}, distinct " +
              " | ".join(f"q{int(s['q']*100)} p={s['prob']:.2f} n={s['n']} ahat={s['contamination']:.3f}" for s in entry["distinct"]), flush=True)
    summary[task] = rec
    json.dump(summary, open(f"{OUT}/v2_summary.json", "w"), indent=1)
print("V2 SELECTIONS DONE")
