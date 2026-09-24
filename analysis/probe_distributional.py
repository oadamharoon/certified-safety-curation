"""Probe 22: WHICH PROPERTY OF A CURATED SELECTION PREDICTS THE CLONE'S COST?

The paper measures that realized contamination explains R^2 = 0.38 of clone cost
(task-demeaned) and score margin lifts it to 0.50. Half the variance is unexplained.
Probe 21 found Ant's loss tail carries no safety information while every other task's
does. This probe computes a battery of candidate statistics of a selection, beyond the
fraction unsafe, and asks which one predicts clone cost, and whether it explains Ant.

Selections: the 14 reconstructible recipe selections per task used by
certified-safety-curation/analysis/margin_expand.py (bc_all, bcsafe, vfilt_return,
vfilt_retbot, and vfilt_matchgt / vfilt_q25 per pipeline seed), matched to archived
clone costs in results_snapshot.json. Score margins are reused from margin_expand.json.

CPU only: nearest-neighbour statistics, no training.
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

import json, os, pickle, sys, time
import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

W = CSC_WORKSPACE
sys.path.insert(0, f"{W}/vlm-with-cpl/new_data")
os.chdir(f"{W}/vlm-with-cpl/new_data")
cfg = yaml.safe_load(open("config.yaml"))
SNAP = json.load(open(f"{W}/iclr2027/data/results_snapshot.json"))
MARG = {r["sel"]: r for r in json.load(
    open(f"{W}/iclr2027/data/review_response/margin_expand.json"))["rows"]}
OUT = f"{W}/runs/probe/distributional.json"
TASKS = [t for t in SNAP if "vfilt_return" in SNAP[t] and not t.endswith("_b")]
if os.environ.get("DP_TASKS"):
    TASKS = os.environ["DP_TASKS"].split(",")
SEED = 0
rng = np.random.default_rng(SEED)
NS = 12000      # states sampled per class for separability / conflict
NC = 4000       # pool-safe states sampled for coverage
print(f"{len(TASKS)} tasks", flush=True)


def sample_states(trajs, idx, n, rng):
    """Uniform sample of up to n states over trajectories idx; returns obs, act, traj id."""
    lens = np.array([len(trajs[i]["observations"]) for i in idx])
    tot = int(lens.sum())
    if tot == 0:
        return None
    pick = rng.choice(tot, size=min(n, tot), replace=False)
    owner = np.repeat(np.arange(len(idx)), lens)
    offs = pick - np.concatenate([[0], np.cumsum(lens)[:-1]])[owner[pick]]
    obs = np.stack([trajs[idx[o]]["observations"][k] for o, k in zip(owner[pick], offs)])
    act = np.stack([trajs[idx[o]]["actions"][k] for o, k in zip(owner[pick], offs)])
    return obs.astype(np.float32), act.astype(np.float32), np.asarray(idx)[owner[pick]]


def nn_other_traj(query_obs, query_tid, ref_obs, ref_tid, k=32):
    """Index into ref of the nearest ref state belonging to a DIFFERENT trajectory."""
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_obs)), n_jobs=16).fit(ref_obs)
    d, j = nn.kneighbors(query_obs)
    out = np.full(len(query_obs), -1)
    dist = np.full(len(query_obs), np.nan)
    for r in range(len(query_obs)):
        ok = np.where(ref_tid[j[r]] != query_tid[r])[0]
        if len(ok):
            out[r] = j[r, ok[0]]
            dist[r] = d[r, ok[0]]
    m = out >= 0
    return out[m], dist[m], m


def features(task, trajs, sel, costs, R, lim, mu, sd, pool_safe_obs, pool_safe_tid,
             pool_safe_self_d, act_sd, rng):
    sel = np.asarray(sel)
    c = costs[sel]
    unsafe = sel[c > lim]
    safe = sel[c <= lim]
    f = {
        "n_kept": int(len(sel)), "kept_frac": float(len(sel) / len(trajs)),
        "contam": float((c > lim).mean()),
        "mean_cost_norm": float(c.mean() / lim),
        "unsafe_excess": float(((c[c > lim] - lim) / lim).mean()) if len(unsafe) else 0.0,
        "safe_margin": float(((lim - c[c <= lim]) / lim).mean()) if len(safe) else 0.0,
        "ret_z": float((R[sel].mean() - R.mean()) / (R.std() + 1e-8)),
    }
    S = sample_states(trajs, safe, NS, rng)
    U = sample_states(trajs, unsafe, NS, rng) if len(unsafe) >= 3 else None
    SAFE_KEYS = ["act_within_safe", "state_nn_within", "sep_auc_lr", "sep_auc_knn",
                 "act_conflict", "state_nn_cross", "unsafe_isolation"]
    if S is None:
        # no safe trajectory kept: every safe-relative statistic is undefined
        f.update({k: np.nan for k in SAFE_KEYS})
    else:
        so, sa, st = S
        so = (so - mu) / sd
    # --- coverage of the pool's safe region by the kept set ---
    K = sample_states(trajs, sel, 2 * NS, rng)
    ko = (K[0] - mu) / sd
    nn = NearestNeighbors(n_neighbors=1, n_jobs=16).fit(ko)
    d_to_kept = nn.kneighbors(pool_safe_obs)[0][:, 0]
    f["cov_ratio"] = float(d_to_kept.mean() / pool_safe_self_d)
    # --- action multimodality inside the kept set (different-trajectory neighbours) ---
    ka, kt = K[1], K[2]
    q = rng.choice(len(ko), size=min(3000, len(ko)), replace=False)
    nn = NearestNeighbors(n_neighbors=24, n_jobs=16).fit(ko)
    _, j = nn.kneighbors(ko[q])
    mm = []
    for r, row in enumerate(j):
        oth = row[kt[row] != kt[q[r]]][:10]
        if len(oth) >= 5:
            mm.append(ka[oth].std(axis=0).mean())
    f["act_multimodal"] = float(np.mean(mm) / act_sd) if mm else np.nan
    if S is None:
        return f
    # --- within-safe action agreement at nearest other-trajectory safe state ---
    j_s, d_s, m_s = nn_other_traj(so, st, so, st)
    d_within = np.linalg.norm(sa[m_s] - sa[j_s], axis=1)
    f["act_within_safe"] = float(d_within.mean() / act_sd)
    f["state_nn_within"] = float(np.nanmean(d_s))
    if U is None:
        f.update({"sep_auc_lr": np.nan, "sep_auc_knn": np.nan, "act_conflict": np.nan,
                  "state_nn_cross": np.nan, "unsafe_isolation": np.nan})
        return f
    uo, ua, ut = U
    uo = (uo - mu) / sd
    # --- state-space separability of kept-unsafe vs kept-safe, trajectory-grouped CV ---
    X = np.concatenate([so, uo]); y = np.r_[np.zeros(len(so)), np.ones(len(uo))]
    g = np.r_[st, ut]
    n_groups = len(np.unique(g))
    if n_groups >= 5:
        p_lr = np.zeros(len(y)); p_kn = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            lr = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr])
            p_lr[te] = lr.predict_proba(X[te])[:, 1]
            kn = KNeighborsClassifier(n_neighbors=25, n_jobs=16).fit(X[tr], y[tr])
            p_kn[te] = kn.predict_proba(X[te])[:, 1]
        f["sep_auc_lr"] = float(roc_auc_score(y, p_lr))
        f["sep_auc_knn"] = float(roc_auc_score(y, p_kn))
    else:
        f["sep_auc_lr"] = f["sep_auc_knn"] = np.nan
    # --- action conflict: unsafe state's action vs nearest kept-safe state's action ---
    j_c, d_c, m_c = nn_other_traj(uo, ut, so, st)
    d_cross = np.linalg.norm(ua[m_c] - sa[j_c], axis=1)
    f["act_conflict"] = float(d_cross.mean() / max(d_within.mean(), 1e-8))
    f["state_nn_cross"] = float(np.nanmean(d_c))
    # how isolated are unsafe states from safe ones, relative to safe-safe spacing
    f["unsafe_isolation"] = float(np.nanmean(d_c) / max(np.nanmean(d_s), 1e-8))
    return f


rows = json.load(open(OUT)) if os.path.exists(OUT) and os.environ.get("DP_RESUME") else []
done = {r["task"] for r in rows}
t0 = time.time()
for task in TASKS:
    if task in done:
        continue
    lim = 20 if "velocity" in task else 25
    trajs = pickle.load(open(cfg["tasks"][task]["data_pickle"], "rb"))
    costs = np.array([float(np.sum(t["costs"])) for t in trajs])
    R = np.array([float(np.sum(t["rewards"])) for t in trajs])
    safe_mask = costs <= lim
    allobs = np.concatenate([t["observations"] for t in trajs]).astype(np.float32)
    mu, sd = allobs.mean(0), allobs.std(0) + 1e-6
    act_sd = float(np.concatenate([t["actions"] for t in trajs]).std())
    del allobs
    P = sample_states(trajs, np.where(safe_mask)[0], NC, rng)
    pool_safe_obs = (P[0] - mu) / sd
    _, d_self, _ = nn_other_traj(pool_safe_obs, P[2], pool_safe_obs, P[2])
    pool_safe_self_d = float(np.nanmean(d_self))
    frac = float(safe_mask.mean())
    nsel = max(1, int(round(frac * len(trajs))))
    order_R = np.argsort(R)[::-1]

    def add(sel, cfgname, seed_key=None):
        ent = SNAP[task].get(cfgname, {})
        if not ent:
            return
        if seed_key is not None:
            if seed_key not in ent:
                return
            y = ent[seed_key]["C"] / lim
        else:
            y = float(np.mean([e["C"] for e in ent.values()])) / lim
        key = f"{cfgname}:{seed_key}"
        f = features(task, trajs, sel, costs, R, lim, mu, sd, pool_safe_obs, P[2],
                     pool_safe_self_d, act_sd, rng)
        m = MARG.get(key, {})
        f.update({"task": task, "sel": key, "cost_norm": float(y),
                  "mean_margin": m.get("mean_margin", np.nan),
                  "p10_margin": m.get("p10_margin", np.nan)})
        rows.append(f)

    add(np.arange(len(trajs)), "bc_all")
    add(np.where(safe_mask)[0], "bcsafe")
    add(order_R[:nsel], "vfilt_return")
    add(order_R[::-1][:nsel], "vfilt_retbot")
    # score-ranked selections need the pipeline scores; rebuild them from the margin rows'
    # recipe by re-scoring is GPU work, so reuse the archived kept sets when present
    import torch
    from model.policy import VEnsemble
    obs_dim = trajs[0]["observations"].shape[1]
    for s in range(5):
        ck_path = f"outputs/{task}/v_ensemble_pess_seed{s}.pt"
        if not os.path.exists(ck_path):
            continue
        ens = VEnsemble(obs_dim, 256, K=3)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ens.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        ens.eval()
        g = np.zeros(len(trajs))
        with torch.no_grad():
            for i, t in enumerate(trajs):
                o = torch.as_tensor(t["observations"], dtype=torch.float32)
                g[i] = ens(o).mean().item()
        top = np.argsort(g)[::-1]
        add(top[:nsel], "vfilt_matchgt", str(s))
        add(top[:max(1, int(round(0.25 * len(trajs))))], "vfilt_q25", str(s))
    print(f"{task}: rows {len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(OUT, "w"), indent=1, default=float)

print(f"wrote {OUT} with {len(rows)} rows", flush=True)
