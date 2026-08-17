"""
Download RNA-FM pretrained weights from Hugging Face.

The official CUHK CDN (proj.cse.cuhk.edu.hk) frequently returns HTTP 403.
This script downloads from Hugging Face instead.

Usage
-----
    python -m physrna_filter.data.download_rnafm_weights

Then set (optional):
    set RNAFM_CHECKPOINT=physrna_filter\\data\\rnafm_weights\\RNA-FM_pretrained.pth
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

DEST_DIR = Path(__file__).parent / "rnafm_weights"
FILENAME = "RNA-FM_pretrained.pth"

URLS = [
    "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth",
    "https://hf-mirror.com/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth",
]


def download_rnafm_weights(dest_dir: Path | None = None) -> Path:
    dest_dir = dest_dir or DEST_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / FILENAME

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest

    tmp = dest.with_suffix(".pth.downloading")

    for url in URLS:
        try:
            print(f"Downloading from {url} ...")
            print("(~1.2 GB — this may take several minutes)")
            urlretrieve(url, tmp)  # noqa: S310
            if tmp.stat().st_size < 1_000_000:
                tmp.unlink(missing_ok=True)
                continue
            tmp.rename(dest)
            print(f"Saved to {dest}")
            return dest
        except (HTTPError, URLError, OSError) as e:
            print(f"  failed: {e}")
            tmp.unlink(missing_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        print("Trying huggingface_hub ...")
        path = hf_hub_download(
            repo_id="cuhkaih/rnafm",
            filename=FILENAME,
            local_dir=str(dest_dir),
        )
        print(f"Saved to {path}")
        return Path(path)
    except Exception as e:
        print(f"huggingface_hub failed: {e}", file=sys.stderr)
        raise SystemExit(
            "Could not download RNA-FM weights.\n"
            "Manual steps:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli download cuhkaih/rnafm RNA-FM_pretrained.pth "
            f"--local-dir {dest_dir}\n"
            "Then set RNAFM_CHECKPOINT to the downloaded file."
        ) from e


def main():
    path = download_rnafm_weights()
    print(f"\nSet environment variable:")
    print(f"  RNAFM_CHECKPOINT={path.resolve()}")


if __name__ == "__main__":
    main()
