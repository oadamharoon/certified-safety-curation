"""Environment / rendering helpers.

Safety-Gymnasium does *not* expose a public ``set_state`` to teleport the
simulator, so we render by replaying actions from ``env.reset()``. This
gives a visually plausible (deterministic given the seed) re-roll of the
trajectory; absolute pixel-equivalence with the dataset is not required
for VLM preference labelling.
"""
from __future__ import annotations

import os
from io import BytesIO
from typing import Dict, Iterable, List, Sequence

# Render off-screen by default (cluster-friendly).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image


def make_render_env(env_name: str, seed: int = 0, camera_id: int | None = None):
    """Create a SafetyGym env in rgb_array mode.

    camera_id selects the MuJoCo camera used for rendering. For
    safety-gymnasium envs the ids are:
        0 = fixednear   1 = fixedfar   2 = vision (egocentric)   3 = track (top-down following)
    Pass None to keep the env's default (typically `vision`).
    """
    import gymnasium as gym
    try:
        import dsrl  # noqa: F401  registers offline ids; harmless for online ones
        if hasattr(__import__("dsrl"), "register_envs"):
            __import__("dsrl").register_envs()
    except Exception:
        pass
    make_kwargs = {"render_mode": "rgb_array"}
    if camera_id is not None:
        make_kwargs["camera_id"] = int(camera_id)
    env = gym.make(env_name, **make_kwargs)
    env.reset(seed=seed)
    _patch_visibility(env)
    return env


def _patch_visibility(env) -> None:
    # Safety-gymnasium puts hazards (group 3), vases (4), goal (9), etc. into
    # non-zero MuJoCo geom groups. Empirically, only group 0 geoms render
    # robustly through the OffScreenViewer pipeline — even though
    # `vopt.geomgroup[:] = 1` says groups 0–5 are visible, geoms left in
    # groups 1–5 come out as a few stray pixels (depth/blend ordering against
    # the floor in group 0). Remapping everything non-floor to group 0
    # restores the documentation rendering. We deliberately do NOT touch
    # rgba alpha: hazards (alpha 0.25) and the CarButton goal halo
    # (alpha 0.1) are translucent on purpose — that's where the lavender
    # appearance of hazards in the docs comes from (blue alpha-blended over
    # the beige floor).
    import mujoco
    m = env.unwrapped.task.model
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if m.geom_group[i] != 0 and name not in ("floor",):
            m.geom_group[i] = 0


def replay_episode_frames(env, actions: np.ndarray, seed: int) -> List[np.ndarray]:
    """Reset env then step through `actions`, capturing one frame per step.

    Returns a list of length ``len(actions)`` of HxWx3 uint8 arrays.
    """
    env.reset(seed=seed)
    _patch_visibility(env)
    frames: List[np.ndarray] = []
    for act in actions:
        env.step(np.asarray(act, dtype=np.float32))
        frames.append(env.render())
    return frames


def _uniform_offsets(length: int, n_frames: int) -> List[int]:
    """PLARE-style: first/middle/last for n=3, evenly spaced for larger n."""
    if n_frames == 1:
        return [length // 2]
    return np.linspace(0, length - 1, n_frames, dtype=int).tolist()


def _cost_aware_offsets(costs: np.ndarray, n_frames: int) -> List[int]:
    """Stratify by cost percentile, then apply the same temporal-uniform
    primitive used by `_uniform_offsets` *within* each cost-level bin.

    Algorithm:
      1. Target an equally-spaced cost percentile for each of the n_frames
         slots (e.g. [0, 25, 50, 75, 100]% for n=5).
      2. Group those targets by their resolved cost value. For continuous
         costs this is one slot per level; for binary safety-gym costs
         several slots collapse to the same level (e.g. {0: 3, 1: 2}).
      3. For each cost level, collect the timesteps whose cost matches it
         most closely, then pick the allocated number of frames at evenly-
         spaced positions across that bin — the same rule as the uniform
         method, just applied within each cost bin.

    Binary-cost envs → safe/unsafe mix proportional to the segment's
    safe/unsafe ratio, with frames temporally spread within each side.
    Continuous-cost envs → graduated cost levels, one frame per level.
    Fully-safe segments (all zeros) → degenerate single bin spanning the
    whole segment, identical to uniform sampling.
    """
    from collections import Counter

    length = int(len(costs))
    if length <= n_frames:
        return list(range(length))
    if np.max(costs) <= 0:
        return _uniform_offsets(length, n_frames)

    targets = np.percentile(costs, np.linspace(0, 100, n_frames))
    level_alloc = Counter(targets.tolist())   # {cost_level: n_slots}

    chosen: List[int] = []
    used: set = set()
    for level, n_alloc in level_alloc.items():
        diffs = np.abs(costs - level)
        candidates = np.where(diffs == diffs.min())[0]
        bin_size = len(candidates)
        if bin_size >= n_alloc:
            picks = candidates[np.linspace(0, bin_size - 1, n_alloc, dtype=int)]
        else:
            picks = candidates
        for p in picks:
            if int(p) not in used:
                chosen.append(int(p))
                used.add(int(p))

    while len(chosen) < n_frames:
        for i in range(length):
            if i not in used:
                chosen.append(int(i))
                used.add(int(i))
                if len(chosen) >= n_frames:
                    break
    return sorted(chosen[:n_frames])


def render_segments_grouped_by_traj(
    env_name: str,
    trajectories: List[Dict],
    segments: List[Dict],
    frame_indices_per_seg: int = 3,
    base_seed: int = 0,
    camera_id: int | None = None,
    frame_selection: str = "uniform",
) -> Dict[int, List[Image.Image]]:
    """For each segment id, render `frame_indices_per_seg` key frames.

    frame_selection:
        "uniform"    — evenly spaced over time (PLARE-style: first/middle/last
                       for n=3; linspace for larger n).
        "cost_aware" — stratified by the segment's cost percentiles so the
                       VLM sees a mix of low / medium / high cost frames
                       (collapses to uniform for fully-safe segments).
    """
    if frame_selection not in ("uniform", "cost_aware"):
        raise ValueError(f"Unknown frame_selection={frame_selection!r}")

    # Group seg ids by trajectory and remember required end indices.
    by_traj: Dict[int, List[Dict]] = {}
    for seg in segments:
        by_traj.setdefault(seg["traj_id"], []).append(seg)

    env = make_render_env(env_name, seed=base_seed, camera_id=camera_id)
    out: Dict[int, List[Image.Image]] = {}

    for tidx, segs in by_traj.items():
        traj = trajectories[tidx]
        max_end = max(s["end"] for s in segs)
        actions = traj["actions"][:max_end]
        # Prefer the seed that was actually used at collection time; fall
        # back to a deterministic per-trajectory seed otherwise.
        seed = int(traj["seed"]) if "seed" in traj else int(base_seed + tidx)
        frames = replay_episode_frames(env, actions, seed=seed)
        for seg in segs:
            length = seg["end"] - seg["start"]
            if frame_selection == "cost_aware":
                offsets = _cost_aware_offsets(np.asarray(seg["costs"]), frame_indices_per_seg)
            else:
                offsets = _uniform_offsets(length, frame_indices_per_seg)
            # Store JPEG-compressed bytes -- ~15x smaller than raw PIL Images.
            key_frames = []
            for o in offsets:
                buf = BytesIO()
                Image.fromarray(frames[seg["start"] + o]).convert("RGB").save(
                    buf, format="JPEG", quality=85)
                key_frames.append(buf.getvalue())
            out[seg["seg_id"]] = key_frames

    env.close()
    return out
