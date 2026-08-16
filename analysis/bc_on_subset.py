"""BC on an explicit kept-index selection (identical-selection parity arm).

Env: SAFETY_VLM_TASK, KEPT_JSON (path), SEED_OVERRIDE, OUT_TAG.
Saves bc_<OUT_TAG>_policy.pt in the task output_dir (04p-compatible format).
"""
import json, os, pickle, sys
import numpy as np
import torch

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
sys.path.insert(0, REPO)
os.chdir(REPO)
from utils.common import load_cfg, set_seed
from model.policy import GaussianPolicy
from model.train import bc_pretrain

cfg = load_cfg()
if "SEED_OVERRIDE" in os.environ:
    cfg["seed"] = int(os.environ["SEED_OVERRIDE"])
seed = int(cfg["seed"])
set_seed(seed)
out_tag = os.environ["OUT_TAG"]
with open(cfg["data_pickle"], "rb") as f:
    trajs = pickle.load(f)
with open(os.environ["KEPT_JSON"]) as f:
    kept = json.load(f)["kept"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("RETURN_TOPHALF"):
    # hard gated-R50 rule on the identical selection: keep top half by return
    R = np.array([float(np.sum(trajs[i]["rewards"])) for i in kept])
    order = np.argsort(R)[::-1][:max(1, len(kept) // 2)]
    kept = [kept[j] for j in order]
    print(f"[toph] hard top-half: {len(kept)} trajectories", flush=True)
if os.environ.get("RETURN_WEIGHTED"):
    # certified return-weighted cloning: resample kept trajectories with
    # probability proportional to exp(clip(z(R), -c, c)); same dataset size.
    clip_c = float(os.environ.get("WEIGHT_CLIP", "2.0"))
    R = np.array([float(np.sum(trajs[i]["rewards"])) for i in kept])
    z = (R - R.mean()) / (R.std() + 1e-8)
    logw = np.clip(z, -clip_c, clip_c)
    if os.environ.get("MARGIN_JSON"):
        # margin regularizer: lambda * clip(z(V-score margin), +/-2)
        lam = float(os.environ.get("MARGIN_LAMBDA", "1.0"))
        gmap = json.load(open(os.environ["MARGIN_JSON"]))
        m = np.array([float(gmap[str(i)]) for i in kept])
        zm = (m - m.mean()) / (m.std() + 1e-8)
        logw = logw + lam * np.clip(zm, -2.0, 2.0)
        print(f"[mwbc] margin term: lambda={lam}", flush=True)
    w = np.exp(logw)
    p_i = w / w.sum()
    rng = np.random.default_rng(10_000 + seed)
    kept = [kept[j] for j in rng.choice(len(kept), size=len(kept),
                                        replace=True, p=p_i)]
    print(f"[wbc] return-weighted resample: eff trajectories "
          f"{len(set(kept))}/{len(p_i)} unique", flush=True)
obs = torch.cat([torch.as_tensor(trajs[i]["observations"], dtype=torch.float32)
                 for i in kept])
act = torch.cat([torch.as_tensor(trajs[i]["actions"], dtype=torch.float32)
                 for i in kept])
obs_dim, act_dim = obs.shape[1], act.shape[1]
print(f"BC on {len(kept)} kept trajs, {len(obs)} transitions, seed {seed}", flush=True)
policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden_dim"]).to(device)
bc_pretrain(policy, obs, act, batch_size=cfg["batch_size"],
            epochs=int(cfg.get("bc_only_epochs", 100)),
            lr=cfg["learning_rate"], device=device,
            log_every=cfg["log_every"], wandb_run=None)
out_p = os.path.join(cfg["output_dir"], f"bc_{out_tag}_policy.pt")
torch.save({"obs_dim": obs_dim, "act_dim": act_dim,
            "hidden_dim": cfg["hidden_dim"],
            "state_dict": policy.state_dict()}, out_p)
print(f"Saved -> {out_p}", flush=True)
