"""Step 01: extract fixed-length segments and filter out idle ones."""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402
from utils.segment_utils import extract_segments, filter_active_segments  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config yaml")
    args, _ = ap.parse_known_args()  # allow --task to pass through
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    with open(cfg["data_pickle"], "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories.")

    segments = extract_segments(trajectories, cfg["segment_length"], cfg["segment_stride"])
    print(f"Extracted {len(segments)} segments (length={cfg['segment_length']}, "
          f"stride={cfg['segment_stride']}).")

    active = filter_active_segments(
        segments,
        min_total_reward=cfg["min_total_reward"],
        drop_zero=cfg["drop_zero_segments"],
    )
    print(f"After filtering: {len(active)} active segments "
          f"({100.0 * len(active) / max(1, len(segments)):.1f}% kept).")

    # Optional cap so we don't render an enormous corpus.
    cap = cfg.get("max_active_segments")
    if cap and len(active) > cap:
        import numpy as np
        rng = np.random.default_rng(cfg["seed"])
        keep = rng.choice(len(active), size=cap, replace=False)
        active = [active[i] for i in sorted(keep.tolist())]
        # re-stamp seg_id for stable indexing downstream
        for new_id, seg in enumerate(active):
            seg["seg_id"] = new_id
        print(f"Subsampled active segments to {len(active)} (cap={cap}).")

    out_dir = cfg["output_dir"]
    with open(os.path.join(out_dir, "segments_raw.pkl"), "wb") as f:
        pickle.dump(segments, f)
    with open(os.path.join(out_dir, "active_segments.pkl"), "wb") as f:
        pickle.dump(active, f)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
