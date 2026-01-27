import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from config import FACES_DIR, CATALOG_CSV


def main():
    rows = []

    for label_name, label_value in [("real", 1), ("deepfakes", 0)]:
        root = FACES_DIR / label_name
        if not root.exists():
            continue

        for video_dir in root.iterdir():
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name

            for img_path in video_dir.glob("*.jpg"):
                rows.append({
                    "path": str(img_path),
                    "label": label_value,
                    "video_id": video_id,
                    "manipulation": label_name
                })

    df = pd.DataFrame(rows)
    CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_CSV, index=False)

    print(f"Catalog written to {CATALOG_CSV}")
    print(df.head())
    print("Total samples:", len(df))


if __name__ == "__main__":
    main()
