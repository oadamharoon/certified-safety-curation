"""MetaDrive adapter (photorealistic driving env).

Isolated from the safety-gym path in env_utils.py because MetaDrive uses
Panda3D, which needs its headless GL backend selected *before* the engine
initializes. Panda3D's EGL pipe (libp3headlessgl) renders on the GPU via
/dev/nvidia* — the same path MuJoCo uses — so no X server / Xvfb / DRI
render-group is required.

The DSRL MetaDrive hdf5 datasets store no scenario seeds, so offline
trajectories cannot be re-rendered faithfully. Instead we train a BC policy
on the hdf5 and roll it out here with *known* seeds, then render those.

Observation is the standard 259-dim MetaDrive vector (240 lidar + ego state),
matching the DSRL data; the RGB camera is added only for rendering and is not
part of the agent observation.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Chase camera pose relative to the ego vehicle origin (behind + above, slight
# downward pitch). Matches the "view_2_chase" comparison render.
CHASE_POSITION: Tuple[float, float, float] = (0.0, -8.0, 3.5)
CHASE_HPR: Tuple[float, float, float] = (0.0, -8.0, 0.0)

# Per-task MetaDrive scenario config. Map / traffic density mirror the DSRL
# offline_metadrive definitions (e.g. EasySparseEnv: map "SCS", traffic 0.1,
# start_seed 0). num_scenarios sets the inclusive seed range MetaDrive
# accepts; bumped from DSRL's 1 to 1000 so collection can use a distinct seed
# per episode (each seed is a different procedural variant of the same map
# pattern — gives behavioral diversity for the preference dataset).
TASK_CONFIGS: Dict[str, Dict] = {
    # Legacy task configs used in Phase 5-8 (non-DSRL conventions). Kept for
    # backward compatibility with already-collected trajectories. New work
    # should use the dsrl_* variants which match DSRL/FSRL published settings.
    "easy_sparse":   {"map": "SCS", "traffic_density": 0.10, "start_seed": 0,   "num_scenarios": 1000},
    "easy_mean":     {"map": "SCS", "traffic_density": 0.20, "start_seed": 0,   "num_scenarios": 1000},
    "easy_dense":    {"map": "SCS", "traffic_density": 0.30, "start_seed": 0,   "num_scenarios": 1000},
    # medium_* originally inherited DSRL's start_seed=100 convention, but our
    # collection script uses raw ep_idx 0..299 as scenario seeds. With start_seed=100
    # those would AssertionError. Use start_seed=0 here so the seed window is
    # [0, num_scenarios) and our seeds 0..299 (main) + 800..949 (calibration pilots)
    # all land inside it. The actual map type and traffic density (the things that
    # define "medium" difficulty) are unaffected — start_seed just shifts the
    # procedural-generation offset and any value in [0, num_scenarios-100) is fine.
    "medium_sparse": {"map": "XST", "traffic_density": 0.10, "start_seed": 0, "num_scenarios": 1000},
    "medium_mean":   {"map": "XST", "traffic_density": 0.20, "start_seed": 0, "num_scenarios": 1000},
    "medium_dense":  {"map": "XST", "traffic_density": 0.30, "start_seed": 0, "num_scenarios": 1000},

    # DSRL-convention task configs (Phase 9+). Match traffic densities, maps,
    # and start_seeds from dsrl.offline_metadrive.gym_envs:
    # densities 0.10/0.15/0.20 = sparse/mean/dense (NOT 0.10/0.20/0.30);
    # maps SCS/XST/TRO = easy/medium/hard; start_seed 0/100/200 = easy/medium/hard.
    # num_scenarios overridden from DSRL's 1 to 1000 so collection can run with
    # distinct seeds per episode (DSRL itself only used scenario 0 for its 1M
    # samples; our online collection needs procedural diversity).
    "dsrl_easy_sparse":    {"map": "SCS", "traffic_density": 0.10, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_easy_mean":      {"map": "SCS", "traffic_density": 0.15, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_easy_dense":     {"map": "SCS", "traffic_density": 0.20, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_medium_sparse":  {"map": "XST", "traffic_density": 0.10, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_medium_mean":    {"map": "XST", "traffic_density": 0.15, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_medium_dense":   {"map": "XST", "traffic_density": 0.20, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_hard_sparse":    {"map": "TRO", "traffic_density": 0.10, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_hard_mean":      {"map": "TRO", "traffic_density": 0.15, "start_seed": 0,   "num_scenarios": 1000},
    "dsrl_hard_dense":     {"map": "TRO", "traffic_density": 0.20, "start_seed": 0,   "num_scenarios": 1000},
}

def _ensure_egl() -> None:
    """Select Panda3D's headless EGL backend (NVIDIA, via /dev/nvidia*).

    Done at import time so the PRC config is registered before any MetaDrive /
    Panda3D graphics engine is created — otherwise Panda3D defaults to the
    GLX (X11) pipe, which on a headless box means slow Mesa software rendering.
    """
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "load-display p3headlessgl")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "notify-level-display error")


# Register the EGL backend as soon as this module is imported.
_ensure_egl()


def make_metadrive_env(task_key: str, image_size: Tuple[int, int] = (480, 360),
                       for_render: bool = False, nonterminal: bool = False,
                       dsrl: bool = False):
    """Create a MetaDrive env. Three configurations.

    MetaDrive ties the graphics engine to the observation type, so we expose
    ``for_render`` to switch between vector obs (collection/eval, fast) and
    offscreen image observations (replay for VLM frames). The cost/termination
    layer is independently selected by ``dsrl`` and ``nonterminal``.

    * ``dsrl=True`` (Phase 9+, the only config comparable to DSRL/PReSa):
      Uses :class:`SafeMetaDriveEnv` as the base + an FSRL-style step wrapper
      that adds per-step velocity_cost = max(0, 0.01·(v − 10)) and
      proximity_cost = 0.01·max(0, 0.1 − min(lidar))². Termination on
      ``arrive_dest`` or ``crash`` or ``out_of_road``; truncation on
      ``max_step``. Mirrors :class:`dsrl.offline_metadrive.gym_envs.
      SafeMetaDriveEnv_FSRL` exactly so trajectories collected here use the
      same cost structure as DSRL's published benchmarks. The constraint
      thresholds 10/20/40 from DSRL are calibrated against this configuration.

    * ``nonterminal=True`` (LEGACY, Phase 5-8): default ``MetaDriveEnv`` with
      all done-on-violation flags disabled. Cost = 1 per step the vehicle is
      off-road or crashing, no per-step velocity/proximity cost. Episodes
      can incur enormous cost tails because nothing truncates. **Not
      DSRL-comparable** — the off-road accumulation is unbounded.

    * Default (``dsrl=False`` and ``nonterminal=False``, LEGACY): default
      ``MetaDriveEnv`` with all done flags at default values (True), so any
      single violation terminates and cost ∈ {0, 1}. Also not DSRL-comparable.

    ``dsrl=True`` and ``nonterminal=True`` are mutually exclusive.
    """
    if dsrl and nonterminal:
        raise ValueError("dsrl=True and nonterminal=True are mutually exclusive")
    if task_key not in TASK_CONFIGS:
        raise KeyError(f"Unknown metadrive task '{task_key}'. "
                       f"Known: {list(TASK_CONFIGS)}")
    tc = TASK_CONFIGS[task_key]

    if dsrl:
        return _make_dsrl_env(tc, image_size, for_render)

    cfg = dict(
        num_scenarios=int(tc.get("num_scenarios", 1000)),
        start_seed=int(tc["start_seed"]),
        traffic_density=float(tc["traffic_density"]),
        map=tc["map"],
        accident_prob=0.0,
        horizon=1000,
        use_render=False,
        interface_panel=[],
    )
    if nonterminal:
        cfg["out_of_road_done"] = False
        cfg["crash_vehicle_done"] = False
        cfg["crash_object_done"] = False
        cfg["on_continuous_line_done"] = False
    if for_render:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        cfg["image_observation"] = True
        cfg["sensors"] = {"rgb_camera": (RGBCamera, image_size[0], image_size[1])}
        cfg["vehicle_config"] = {"image_source": "rgb_camera"}
    else:
        cfg["image_observation"] = False

    from metadrive.envs.metadrive_env import MetaDriveEnv
    return MetaDriveEnv(cfg)


def _make_dsrl_env(tc: Dict, image_size: Tuple[int, int], for_render: bool):
    """Build a DSRL/FSRL-compatible env.

    Concretely: SafeMetaDriveEnv base (per-step crash/out_of_road costs,
    crashes do NOT auto-terminate) + an FSRL step wrapper adding velocity
    and proximity costs plus DSRL-style termination semantics. Mirrors
    dsrl.offline_metadrive.gym_envs.SafeMetaDriveEnv_FSRL.
    """
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
    from metadrive.manager.traffic_manager import TrafficMode

    cfg = dict(
        num_scenarios=int(tc.get("num_scenarios", 1000)),
        start_seed=int(tc["start_seed"]),
        traffic_density=float(tc["traffic_density"]),
        map=tc["map"],
        accident_prob=0.0,
        horizon=1000,
        use_render=False,
        interface_panel=[],
        # ---- FSRL defaults (from SafeMetaDriveEnv_FSRL.default_config) ----
        # NOTE: idm_target_speed/idm_acc_factor/idm_deacc_factor from FSRL
        # are NOT included here — they don't exist in the installed
        # MetaDrive config schema and FSRL injects them via
        # `allow_add_new_key=True`. They control IDM-driven traffic
        # vehicle behavior; omitting them means traffic uses MetaDrive's
        # defaults. The other DSRL-relevant knobs below still take effect.
        crash_vehicle_penalty=0.0,
        random_traffic=True,
        traffic_mode=TrafficMode.Hybrid,
    )
    if for_render:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        cfg["image_observation"] = True
        cfg["sensors"] = {"rgb_camera": (RGBCamera, image_size[0], image_size[1])}
        cfg["vehicle_config"] = {"image_source": "rgb_camera"}
    else:
        cfg["image_observation"] = False

    return _DsrlFsrlWrapper(SafeMetaDriveEnv(cfg))


class _DsrlFsrlWrapper:
    """Per-step cost & termination wrapper around SafeMetaDriveEnv.

    Reproduces ``dsrl.offline_metadrive.gym_envs.SafeMetaDriveEnv_FSRL``:

      velocity_cost  = max(0, 0.01 · (velocity − 10))      # speeding > 10 m/s
      proximity_cost = 0.01 · max(0, 0.1 − min(lidar))²    # vehicles too close
      cost          += velocity_cost + proximity_cost
      out_of_road_cost = self.config["out_of_road_cost"] if out_of_road else 0
      crash_cost       = self.config["crash_vehicle_cost"] if crash      else 0
      terminated = arrive_dest OR crash OR out_of_road
      truncated  = max_step

    Returns the gymnasium 5-tuple. SafeMetaDriveEnv.step still returns the
    4-tuple (obs, reward, done, info) — we discard `done` and recompute
    terminated/truncated from `info` fields.

    Attribute access (engine, agent, current_seed, config, ...) forwards
    transparently via ``__getattr__``.
    """

    def __init__(self, env):
        # Use object.__setattr__ to avoid triggering __getattr__/__setattr__
        # recursion (we override __getattr__ below).
        object.__setattr__(self, "_env", env)

    def step(self, actions):
        # SafeMetaDriveEnv (default base) returns 4-tuple (o, r, done, info).
        ret = self._env.step(actions)
        if len(ret) == 4:
            o, r, _d, i = ret
        else:
            # Newer MetaDrive could return gymnasium 5-tuple; be defensive.
            o, r, _term, _trunc, i = ret
        velocity = i.get("velocity", 0.0)
        velocity_cost = max(0.0, 1e-2 * (velocity - 10.0))
        # The last 240 lidar dims correspond to other-vehicle distances in
        # MetaDrive's 259-dim observation; smaller values = closer vehicles.
        proximity_cost = 0.0
        try:
            min_lidar = float(np.min(o[-240:]))
            proximity_cost = 1e-2 * max(0.0, 0.1 - min_lidar) ** 2
        except Exception:
            pass

        i["velocity_cost"] = velocity_cost
        i["proximity_cost"] = proximity_cost
        i["cost"] = float(i.get("cost", 0.0)) + velocity_cost + proximity_cost
        i["out_of_road_cost"] = (
            self._env.config["out_of_road_cost"] if i.get("out_of_road", False) else 0
        )
        i["crash_cost"] = (
            self._env.config["crash_vehicle_cost"] if i.get("crash", False) else 0
        )

        truncated = bool(i.get("max_step", False))
        terminated = bool(
            i.get("arrive_dest", False)
            or i.get("crash", False)
            or i.get("out_of_road", False)
        )
        return o, r, terminated, truncated, i

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        # Forward all other attribute access to the wrapped env. (Avoid an
        # infinite loop: this only fires when `name` isn't found on `self`.)
        return getattr(self._env, name)


def chase_frame(env) -> np.ndarray:
    """Grab a single chase-view RGB frame (HxWx3 uint8) from the env."""
    cam = env.engine.get_sensor("rgb_camera")
    arr = np.asarray(cam.perceive(
        to_float=False,
        new_parent_node=env.agent.origin,
        position=CHASE_POSITION,
        hpr=CHASE_HPR,
    ))
    if arr.ndim == 4:        # (H, W, 3, stack) -> last
        arr = arr[..., -1]
    # MetaDrive returns BGR; flip to RGB.
    return arr.astype(np.uint8)[..., ::-1]


def replay_metadrive_frames(env, actions: np.ndarray, seed: int) -> List[np.ndarray]:
    """Reset to `seed`, replay `actions`, capture one chase frame per step.

    `env` must be a render env (``make_metadrive_env(..., for_render=True)``).
    The returned image observation is ignored; frames come from the chase cam.
    """
    env.reset(seed=int(seed))
    frames: List[np.ndarray] = []
    for act in actions:
        env.step(np.asarray(act, dtype=np.float32))
        frames.append(chase_frame(env))
    return frames


def render_metadrive_segments_grouped_by_traj(
    task_key: str,
    trajectories: List[Dict],
    segments: List[Dict],
    frame_indices_per_seg: int = 3,
    image_size: Tuple[int, int] = (480, 360),
    frame_selection: str = "uniform",
) -> Dict[int, List[bytes]]:
    """MetaDrive analogue of utils.env_utils.render_segments_grouped_by_traj.

    Efficient: a chase-camera ``perceive()`` is ~10x slower than ``env.step()``,
    so this only renders the specific step indices that some segment needs
    (frame_indices_per_seg per segment), instead of every step in the episode.
    """
    from utils.env_utils import _cost_aware_offsets, _uniform_offsets  # reuse

    if frame_selection not in ("uniform", "cost_aware"):
        raise ValueError(f"Unknown frame_selection={frame_selection!r}")

    # group segments by their parent trajectory
    by_traj: Dict[int, List[Dict]] = {}
    for seg in segments:
        by_traj.setdefault(seg["traj_id"], []).append(seg)

    env = make_metadrive_env(task_key, image_size=image_size, for_render=True)
    out: Dict[int, List[bytes]] = {}

    try:
        for tidx, segs in by_traj.items():
            traj = trajectories[tidx]
            max_end = max(s["end"] for s in segs)
            actions = traj["actions"][:max_end]
            seed = int(traj["seed"])

            # Decide per-segment offsets first, then build the set of global
            # step indices that need rendering across all this traj's segments.
            seg_offsets: Dict[int, List[int]] = {}
            needed: set = set()
            for seg in segs:
                length = seg["end"] - seg["start"]
                if frame_selection == "cost_aware":
                    offsets = _cost_aware_offsets(np.asarray(seg["costs"]),
                                                  frame_indices_per_seg)
                else:
                    offsets = _uniform_offsets(length, frame_indices_per_seg)
                seg_offsets[seg["seg_id"]] = list(offsets)
                for o in offsets:
                    needed.add(seg["start"] + o)

            env.reset(seed=seed)
            frames_by_step: Dict[int, np.ndarray] = {}
            for step_idx, act in enumerate(actions):
                env.step(np.asarray(act, dtype=np.float32))
                if step_idx in needed:
                    frames_by_step[step_idx] = chase_frame(env)

            for seg in segs:
                key_frames: List[bytes] = []
                for o in seg_offsets[seg["seg_id"]]:
                    arr = frames_by_step.get(seg["start"] + o)
                    if arr is None:
                        # Shouldn't happen, but guard against off-by-one.
                        continue
                    buf = BytesIO()
                    Image.fromarray(arr).convert("RGB").save(
                        buf, format="JPEG", quality=85)
                    key_frames.append(buf.getvalue())
                out[seg["seg_id"]] = key_frames
    finally:
        env.close()
    return out
