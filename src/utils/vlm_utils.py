"""Back-compat shim: keep ``from utils.vlm_utils import query_vlm`` working
after refactoring backends into ``utils/vlm_backends/``.
"""
from utils.vlm_backends import query_vlm, build_prompt_from_template  # noqa: F401

__all__ = ["query_vlm", "build_prompt_from_template"]
