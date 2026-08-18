"""Build flat DSRL-format hdf5 subsets from the regenerated certified selections,
so CDT can be retrained on exactly the certified data (A3)."""
import json, os, pickle, sys
import numpy as np, h5py, yaml
D="/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
SEL="/home/omniverse/workspace/safevlmcpl/runs/selections"
os.chdir(D); sys.path.insert(0, D)
cfg=yaml.safe_load(open("config.yaml"))
made=0
for f in sorted(os.listdir(SEL)):
    if not f.endswith("_kept.json"): continue
    tag=f[:-len("_kept.json")]
    task=None
    for t in cfg["tasks"]:
        if tag.startswith(t) and (task is None or len(t)>len(task)): task=t
    if task is None: print("  skip (no task):", tag); continue
    kept=json.load(open(os.path.join(SEL,f)))["kept"]
    trajs=pickle.load(open(cfg["tasks"][task]["data_pickle"],"rb"))
    # next_observations by within-trajectory shift; the final step repeats its
    # own observation, matching the DSRL convention for timeout-terminated data.
    nxt=[]
    for i in kept:
        o=np.asarray(trajs[i]["observations"])
        nxt.append(np.concatenate([o[1:], o[-1:]], axis=0))
    nobs=np.concatenate(nxt)
    obs=np.concatenate([trajs[i]["observations"] for i in kept])
    act=np.concatenate([trajs[i]["actions"] for i in kept])
    rew=np.concatenate([trajs[i]["rewards"] for i in kept])
    cost=np.concatenate([trajs[i]["costs"] for i in kept])
    term=np.zeros(len(obs),dtype=bool); tout=np.zeros(len(obs),dtype=bool)
    p=0
    for i in kept:
        p+=len(trajs[i]["rewards"]); tout[p-1]=True
    out=os.path.join(SEL,f"{tag}.hdf5")
    with h5py.File(out,"w") as h:
        for k,v in (("observations",obs),("next_observations",nobs),
                    ("actions",act),("rewards",rew),
                    ("costs",cost),("terminals",term),("timeouts",tout)):
            h.create_dataset(k,data=v)
    made+=1
    print(f"  {tag}: {len(kept)} trajs, {len(obs)} transitions -> {os.path.basename(out)}", flush=True)
print(f"built {made} hdf5 subsets")
