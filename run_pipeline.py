from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str = "config/pipeline.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def main() -> None:
    cfg = load_config()
    print("Config loaded:", cfg)
    print("Next: implement ingest -> clean -> features -> dataset -> train")

if __name__ == "__main__":
    main()
