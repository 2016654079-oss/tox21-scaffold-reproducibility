"""Download the public Tox21 archive used by the preparation scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tox21.csv.gz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response, args.output.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    print(f"downloaded {URL} -> {args.output}")


if __name__ == "__main__":
    main()
