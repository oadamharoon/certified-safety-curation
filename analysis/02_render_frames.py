"""Step 02: render key frames for active segments by replaying their
parent trajectories.

Frames are stored in a *separate* pickle (``segment_frames.pkl``) so the
main ``active_segments.pkl`` stays small and JSON-friendly.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import load_cfg, set_seed  # noqa: E402
from utils.env_utils import render_segments_grouped_by_traj  # noqa: E402


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
    with open(os.path.join(cfg["output_dir"], "active_segments.pkl"), "rb") as f:
        active = pickle.load(f)
    print(f"Rendering {len(active)} active segments "
          f"using env={cfg['env_name']} ...")

    if cfg.get("data_source") == "metadrive":
        from utils.metadrive_utils import render_metadrive_segments_grouped_by_traj
        size = int(cfg.get("render_image_size", 480))
        frame_map = render_metadrive_segments_grouped_by_traj(
            task_key=cfg["metadrive_task"],
            trajectories=trajectories,
            segments=active,
            frame_indices_per_seg=cfg["frames_per_segment"],
            image_size=(size, int(size * 3 / 4)),  # 4:3 aspect
            frame_selection=cfg.get("frame_selection", "uniform"),
        )
    else:
        frame_map = render_segments_grouped_by_traj(
            env_name=cfg["env_name"],
            trajectories=trajectories,
            segments=active,
            frame_indices_per_seg=cfg["frames_per_segment"],
            base_seed=cfg["seed"],
            camera_id=cfg.get("render_camera_id"),
            frame_selection=cfg.get("frame_selection", "uniform"),
        )

    out_path = os.path.join(cfg["output_dir"], "segment_frames.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(frame_map, f)
    print(f"Saved {len(frame_map)} segment->frames mappings to {out_path}")


if __name__ == "__main__":
    main()
