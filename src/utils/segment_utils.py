"""Segment extraction, filtering and preference-pair sampling.

Designed for binary-cost SafetyGym-style tasks (PointGoal/CarCircle).
The 60-d PointGoal observation is *not* xy, so we cannot use it for
displacement filtering. Instead we filter on per-segment reward / cost.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_segments(
    trajectories: List[Dict],
    segment_length: int,
    segment_stride: int,
) -> List[Dict]:
    """Slice each trajectory into fixed-length segments.

    The stride loop yields all aligned segments [k*stride, k*stride + L).
    We then guarantee that the *trailing* ``segment_length`` steps of every
    trajectory are also covered by appending a final segment
    ``[T - L, T)`` whenever the last aligned segment doesn't reach ``T``.

    This matters for envs where trajectories terminate AT the unsafe event
    (e.g. MetaDrive's out_of_road / collision termination puts the cost
    spike on the LAST step). Without this, only episodes whose length T
    happens to be reached by an aligned segment end include the crash —
    which biases the segment pool heavily toward the safe class.

    The added tail segment may slightly overlap with the previous aligned
    one when T isn't a multiple of the stride; that's acceptable and is
    the price for guaranteeing terminal-step coverage.
    """
    segments: List[Dict] = []
    for tidx, traj in enumerate(trajectories):
        T = len(traj["observations"])
        if T < segment_length:
            continue
        # Pre-compute trajectory-level totals so the preference sampler can
        # filter on "demonstration trajectory was actually good (low cost AND
        # high reward) vs actually bad (high cost)" — not just on segment-
        # level cost. This is what fixes the "cost=0 segment from a doomed
        # trajectory contaminates the preferred pool" problem.
        traj_total_cost   = float(np.sum(traj["costs"]))
        traj_total_reward = float(np.sum(traj["rewards"]))
        starts = list(range(0, T - segment_length + 1, segment_stride))
        # Ensure trailing-segment coverage when stride doesn't divide (T-L).
        if not starts or starts[-1] + segment_length < T:
            starts.append(T - segment_length)
        for start in starts:
            end = start + segment_length
            seg = {
                "seg_id": len(segments),
                "traj_id": tidx,
                "start": int(start),
                "end": int(end),
                "observations": traj["observations"][start:end],
                "actions": traj["actions"][start:end],
                "rewards": traj["rewards"][start:end],
                "costs": traj["costs"][start:end],
                "total_reward": float(np.sum(traj["rewards"][start:end])),
                "total_cost": float(np.sum(traj["costs"][start:end])),
                # parent-trajectory totals (same value for every segment in
                # the same trajectory) — used by trajectory-aware preference
                # filtering in sample_pair_indices.
                "traj_total_cost": traj_total_cost,
                "traj_total_reward": traj_total_reward,
            }
            segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def is_active(seg: Dict, min_total_reward: float, drop_zero: bool) -> bool:
    if seg["total_reward"] < min_total_reward and seg["total_cost"] < 1e-6:
        # idle segment - no progress, no danger
        return False
    if drop_zero and abs(seg["total_reward"]) < 1e-6 and seg["total_cost"] < 1e-6:
        return False
    return True


def filter_active_segments(
    segments: List[Dict],
    min_total_reward: float,
    drop_zero: bool = True,
) -> List[Dict]:
    return [s for s in segments if is_active(s, min_total_reward, drop_zero)]


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------
def sample_pair_indices(
    segments: List[Dict],
    cost_safe_threshold: float,
    cost_contrast_min: float,
    rng: np.random.Generator,
    # NEW: trajectory-level quality filters. When both are > 0, "safe" pool
    # is restricted to segments from fully-safe AND successful trajectories
    # (low trajectory cost, high trajectory reward), and "unsafe" pool to
    # segments from trajectories with substantial total cost. This keeps the
    # preferred pool from being contaminated by cost=0 segments that come
    # from trajectories that later fail (the "calm before the storm"
    # problem on terminal-cost envs).
    traj_safe_max_cost: float = -1.0,    # if >= 0, traj_total_cost must be <= this for safe pool
    traj_safe_min_reward: float = -1.0,  # if >= 0, traj_total_reward must be >= this for safe pool
    traj_unsafe_min_cost: float = -1.0,  # if >= 0, traj_total_cost must be >= this for unsafe pool
) -> Tuple[int, int]:
    """Return indices (i, j) of two contrasting segments."""
    costs = np.array([s["total_cost"] for s in segments])
    safe_mask = costs <= cost_safe_threshold
    unsafe_mask = costs > cost_safe_threshold
    # Optional trajectory-quality filtering
    if traj_safe_max_cost >= 0.0 or traj_safe_min_reward >= 0.0 or traj_unsafe_min_cost >= 0.0:
        traj_cost   = np.array([s.get("traj_total_cost",   0.0) for s in segments])
        traj_reward = np.array([s.get("traj_total_reward", 0.0) for s in segments])
        if traj_safe_max_cost >= 0.0:
            safe_mask &= traj_cost <= traj_safe_max_cost
        if traj_safe_min_reward >= 0.0:
            safe_mask &= traj_reward >= traj_safe_min_reward
        if traj_unsafe_min_cost >= 0.0:
            unsafe_mask &= traj_cost >= traj_unsafe_min_cost
    safe = np.where(safe_mask)[0]
    unsafe = np.where(unsafe_mask)[0]

    if len(safe) and len(unsafe):
        for _ in range(100):
            i = int(rng.choice(safe))
            j = int(rng.choice(unsafe))
            if abs(costs[i] - costs[j]) >= cost_contrast_min:
                return (i, j) if rng.random() < 0.5 else (j, i)

    # Fallback: any pair with sufficient cost contrast
    for _ in range(200):
        idx = rng.choice(len(segments), size=2, replace=False)
        i, j = int(idx[0]), int(idx[1])
        if abs(costs[i] - costs[j]) >= cost_contrast_min:
            return (i, j)

    # Last resort
    idx = rng.choice(len(segments), size=2, replace=False)
    return (int(idx[0]), int(idx[1]))
