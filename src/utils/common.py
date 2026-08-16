"""Shared bootstrap helpers for all pipeline scripts.

Every script in scripts/ should start with:

    from utils.common import load_cfg, set_seed, init_wandb, REPO_ROOT
    cfg = load_cfg()

This guarantees correct working directory, importable packages and
reproducible seeding, regardless of where the script is launched from.
"""
from __future__ import annotations

import os
import sys
import random
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Make the repo root importable so ``from model.X`` and ``from utils.X`` work
# no matter where the script is launched from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from a local .env (ignored by git).
# This lets users keep API keys outside source code / config.yaml.
try:
    from dotenv import load_dotenv
    _env_file = REPO_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    # python-dotenv is optional; fall back to plain os.environ.
    pass


def load_cfg(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load config.yaml and optionally merge a task-specific override block.

    Task selection (in priority order):
      1. ``--task <name>`` on the command line
      2. ``SAFETY_VLM_TASK`` environment variable
      3. No task → use top-level defaults as-is

    Example::

        python scripts/01_segment_and_filter.py --task cargoal1
    """
    if path is None and os.environ.get("SAFETY_VLM_CONFIG"):
        path = os.environ["SAFETY_VLM_CONFIG"]
    cfg_path = pathlib.Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- resolve task name ---
    task_name: Optional[str] = None
    # scan sys.argv for --task <name>
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--task", "-t") and i + 1 < len(argv):
            task_name = argv[i + 1]
            break
        if arg.startswith("--task="):
            task_name = arg.split("=", 1)[1]
            break
    if task_name is None:
        task_name = os.environ.get("SAFETY_VLM_TASK")

    # --- merge task overrides ---
    tasks = cfg.pop("tasks", {}) or {}
    if task_name:
        if task_name not in tasks:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {sorted(tasks.keys())}"
            )
        cfg.update(tasks[task_name])
        cfg["_task"] = task_name
        print(f"[config] task={task_name}")
    else:
        cfg["_task"] = None

    cfg["_repo_root"] = str(REPO_ROOT)
    # Resolve well-known relative paths against repo root.
    for key in ("raw_npz", "data_pickle", "output_dir", "prompt_template"):
        if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
            cfg[key] = str(REPO_ROOT / cfg[key])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def init_wandb(cfg: Dict[str, Any], job_type: str) -> Optional[Any]:
    """Initialise a W&B run if enabled. Returns the run handle or None."""
    if not cfg.get("use_wandb"):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed -> skipping logging. `pip install wandb`")
        return None
    return wandb.init(
        project=cfg.get("wandb_project", "safety-vlm-cpl"),
        name=f"{cfg.get('wandb_run_name', 'run')}-{job_type}",
        entity=cfg.get("wandb_entity"),
        group=cfg.get("wandb_run_name", "run"),
        job_type=job_type,
        config=cfg,
        reinit=True,
    )
