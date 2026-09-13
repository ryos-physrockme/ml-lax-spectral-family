from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2026, 2718, 31415, 4242])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = yaml.safe_load(args.config.read_text())
    train_script = Path(__file__).with_name("train.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_config = dict(base_config)
        run_config["seed"] = int(seed)
        run_config["output_dir"] = str(args.output_root / f"seed_{seed}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)
            config_path = Path(handle.name)

        print(f"\n=== seed {seed} ===", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(train_script), "--config", str(config_path), "--device", args.device],
                check=True,
            )
        finally:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
