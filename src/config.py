"""
Configuration Loader

WHY  : Centralised config avoids hard-coded values scattered across files.
HOW  : Reads config/config.yaml and returns a plain dict.
WHEN : Imported by any module that needs configuration.
WHERE: All environments (dev, staging, production).
WHAT : Returns nested dict matching the YAML structure.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML file. Defaults to config/config.yaml
                     relative to the project root.

    Returns:
        dict with all configuration values.
    """
    if config_path is None:
        # Walk up from src/ to find project root, then config/
        root = Path(__file__).parent.parent
        config_path = root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2))
