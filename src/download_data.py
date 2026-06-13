"""Download and prepare ASCAD datasets."""

import os
import sys
import zipfile
import urllib.request
import socket
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, DATASET_CONFIGS

socket.setdefaulttimeout(60)


class DownloadProgressBar:
    def __init__(self, total, desc="Downloading"):
        self.pbar = tqdm(
            total=total if total > 0 else None,
            unit="B", unit_scale=True, desc=desc
        )

    def __call__(self, block_num, block_size, total_size):
        if self.pbar.total is None and total_size > 0:
            self.pbar.total = total_size
            self.pbar.refresh()
        if total_size > 0:
            self.pbar.total = total_size
        self.pbar.update(block_size)


def download_ascad_v1():
    """Download and extract ASCAD v1 datasets (one ZIP contains all 3 H5 variants)."""
    cfg = DATASET_CONFIGS["ascad_v1_fixed"]
    zip_path = os.path.join(DATA_DIR, cfg["filename"])
    extract_marker = os.path.join(DATA_DIR, cfg["extracted_dir"])

    os.makedirs(DATA_DIR, exist_ok=True)

    # Check which H5 files are missing
    needed_files = [
        os.path.join(DATA_DIR, "ASCAD_data", "ASCAD_databases", "ASCAD.h5"),
        os.path.join(DATA_DIR, "ASCAD_data", "ASCAD_databases", "ASCAD_desync50.h5"),
        os.path.join(DATA_DIR, "ASCAD_data", "ASCAD_databases", "ASCAD_desync100.h5"),
    ]
    all_present = all(os.path.exists(f) for f in needed_files)

    if all_present:
        print(f"[SKIP] All ASCAD v1 H5 files already present.")
        return

    # Download ZIP if needed
    if not os.path.exists(zip_path):
        print(f"Downloading ASCAD v1 fixed-key ZIP (~4.4 GB) ...")
        print(f"  URL: {cfg['url']}")
        try:
            urllib.request.urlretrieve(
                cfg["url"], zip_path,
                reporthook=DownloadProgressBar(total=0, desc="ASCAD v1"))
            print("\nDownload complete.")
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            print("Alternative: download manually from")
            print("  https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return
    else:
        print(f"[SKIP] ZIP already exists at {zip_path}")
        # Verify ZIP integrity
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad = zf.testzip()
                if bad:
                    print(f"[WARN] ZIP corrupt (bad file: {bad}). Re-downloading...")
                    os.remove(zip_path)
                    return download_ascad_v1()
        except Exception:
            print(f"[WARN] Cannot verify ZIP. Re-downloading...")
            os.remove(zip_path)
            return download_ascad_v1()

    # Extract
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, DATA_DIR)
    print("Extraction complete.")


def download_ascad_v2():
    """Download ASCAD v2 extracted H5 from data.gouv.fr."""
    cfg = DATASET_CONFIGS["ascad_v2"]
    h5_path = os.path.join(DATA_DIR, cfg["h5_train"])

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(h5_path):
        # Verify H5 integrity by checking it opens
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                pass
            print(f"[SKIP] ASCAD v2 H5 already at {h5_path}")
            return
        except Exception:
            print(f"[WARN] Existing H5 corrupt, re-downloading...")
            os.remove(h5_path)

    print(f"Downloading ASCAD v2 extracted H5 (~7.2 GB) ...")
    print(f"  URL: {cfg['url']}")
    try:
        urllib.request.urlretrieve(
            cfg["url"], h5_path,
            reporthook=DownloadProgressBar(total=0, desc="ASCAD v2"))
        print("\nDownload complete.")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Manual download: https://object.files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5")
        print(f"Place the H5 file in: {DATA_DIR}")
        if os.path.exists(h5_path):
            os.remove(h5_path)


def check_data():
    """Verify that all datasets are available."""
    import h5py  # noqa: F401

    status = {}
    print("\nDataset status:")
    for name, cfg in DATASET_CONFIGS.items():
        h5_path = os.path.join(DATA_DIR, cfg["h5_train"])
        exists = os.path.exists(h5_path)
        status[name] = exists
        marker = "[OK]" if exists else "[MISSING]"
        print(f"  {marker} {name}: {h5_path}")

    return all(status.values())


if __name__ == "__main__":
    print("=" * 60)
    print("ASCAD Dataset Downloader")
    print("=" * 60)

    download_ascad_v1()
    download_ascad_v2()

    all_ok = check_data()

    if all_ok:
        print("\nAll datasets ready.")
    else:
        print("\nSome datasets are missing. See instructions above.")
