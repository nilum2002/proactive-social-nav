"""Config loading for this directory's scripts. Ground truth itself is built
by mot_benchmark_frog/frog_gt.py (not reimplemented here) -- see config.yaml.
"""
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=DEFAULT_CONFIG):
    """Load config.yaml and make every path absolute, resolved against the
    config file's own directory (not the process cwd)."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(path)
    root = os.path.abspath(os.path.join(cfg_dir, cfg["paths"].get("repo_root", "..")))
    cfg["paths"]["repo_root"] = root
    for k, v in cfg["paths"].items():
        if k not in ("repo_root", "bag"):
            cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(root, v)
    return cfg
