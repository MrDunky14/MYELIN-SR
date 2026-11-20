import urllib.request
import os
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent / "data"

SET5_DIR = BASE_DIR / "Set5"
SET14_DIR = BASE_DIR / "Set14"
TRAIN_DIR = BASE_DIR / "train_hr"

SET5_DIR.mkdir(parents=True, exist_ok=True)
SET14_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Standard Set5 image names
SET5_NAMES = ["baby", "bird", "butterfly", "head", "woman"]

def download_file(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            with open(target_path, "wb") as f:
                f.write(content)
        # Verify it's a valid image
        with Image.open(target_path) as im:
            im.verify()
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if target_path.exists():
            target_path.unlink()
        return False

print("Fetching Set5...")
for i, name in enumerate(SET5_NAMES, 1):
    url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_2/img_{i:03d}_SRF_2_HR.png"
    target = SET5_DIR / f"{name}.png"
    if download_file(url, target):
        print(f"  Downloaded {name}.png ({target.stat().st_size} bytes)")

print("\nFetching Set14...")
for i in range(1, 15):
    url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_2/img_{i:03d}_SRF_2_HR.png"
    target = SET14_DIR / f"set14_{i:02d}.png"
    if download_file(url, target):
        print(f"  Downloaded set14_{i:02d}.png ({target.stat().st_size} bytes)")

print("\nFetching BSD100 for train_hr (50 high-res natural images)...")
downloaded_train = 0
for i in range(1, 51):
    url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/BSD100/image_SRF_2/img_{i:03d}_SRF_2_HR.png"
    target = TRAIN_DIR / f"bsd_{i:03d}.png"
    if download_file(url, target):
        downloaded_train += 1

print(f"\nDone! Set5: {len(list(SET5_DIR.glob('*.png')))} images, Set14: {len(list(SET14_DIR.glob('*.png')))} images, Train HR: {downloaded_train} images.")
