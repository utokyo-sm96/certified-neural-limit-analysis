#!/usr/bin/env python3
"""Download the release assets (trained networks, synthetic plates, FE references) into cnla/data.
Usage: python scripts/download_assets.py [--extra]"""
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/utokyo-sm96/certified-neural-limit-analysis/releases/download/v1.0.0/"
ASSETS = ["cnla-data-v1.0.0.zip"] + (["cnla-networks-extra-v1.0.0.zip"] if "--extra" in sys.argv else [])


def main():
    out = os.path.join(ROOT, "cnla", "data")
    os.makedirs(out, exist_ok=True)
    for name in ASSETS:
        path = os.path.join(out, name)
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URL + name, path)
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        os.remove(path)
    print(f"done: {sorted(f for f in os.listdir(out) if f != 'README.md')}")


if __name__ == "__main__":
    main()
