# downloads the raw data into data/raw, run once by hand

import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_URL, RAW_FILE

RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Downloading", DATA_URL)
urllib.request.urlretrieve(DATA_URL, RAW_FILE)
print("Saved to", RAW_FILE)
