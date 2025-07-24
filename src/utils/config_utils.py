import argparse
import ast
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from numpy.random import default_rng


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file.",
        required=True,
        # default="./configs/config.yaml",
    )

    return parser.parse_args()


def tuple_constructor(loader, node):
    """Constructs a python tuple from a YAML string."""
    try:
        return ast.literal_eval(node.value)
    except (ValueError, SyntaxError):
        return node.value


yaml.add_constructor("!tuple", tuple_constructor, Loader=yaml.SafeLoader)


def tuple_representer(dumper, data):
    """Represents a python tuple as a !tuple tagged YAML string."""
    return dumper.represent_scalar("!tuple", str(data))


yaml.add_representer(tuple, tuple_representer, Dumper=yaml.SafeDumper)


def load_config() -> Dict[str, Any]:
    """Load YAML configuration file."""
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "fit_params" in config:
        if config.get("run_id") is None:
            config["run_id"] = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

        file_path = Path(f"{config['output_dir']}/{config['experiment']}/{config['run_id']}/config.yaml")
        file_path.parent.mkdir(parents=True, exist_ok=False)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    return config


def handle_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    rs_numpy = default_rng(seed)
    return rs_numpy
