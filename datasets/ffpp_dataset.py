import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import CATALOG_CSV, DATA_DIR
from utils.preprocessing import get_transforms


class FFPPDataset(Dataset):
    def __init__(self, split="train", split_ratio=(0.7, 0.15, 0.15)):
        df = pd.read_csv(CATALOG_CSV)

        # video-wise split
        video_ids = sorted(df["video_id"].unique())
        n = len(video_ids)

        n_train = int(split_ratio[0] * n)
        n_val = int(split_ratio[1] * n)

        if split == "train":
            selected = video_ids[:n_train]
        elif split == "val":
            selected = video_ids[n_train:n_train + n_val]
        else:
            selected = video_ids[n_train + n_val:]

        self.df = df[df["video_id"].isin(selected)].reset_index(drop=True)
        # Fix paths that point to a different machine (e.g., 'E:\\project\\media-verification\\data...')
        from pathlib import Path

        def fix_path(p):
            p = str(p)
            candidate = Path(p)
            if candidate.exists():
                return str(candidate)

            # common old-root patterns (Windows paths from the other laptop)
            old_roots = [
                r"E:\\project\\media-verification\\data",
                r"E:/project/media-verification/data",
                r"e:\\project\\media-verification\\data",
            ]
            for old in old_roots:
                if p.startswith(old):
                    new_p = str(Path(str(DATA_DIR)) / p[len(old) + 1 :].replace('\\', '/'))
                    if Path(new_p).exists():
                        return new_p

            # fallback: try to find the filename under DATA_DIR
            name = Path(p).name
            try:
                found = next(Path(DATA_DIR).rglob(name))
                return str(found)
            except StopIteration:
                return None

        self.df["path"] = self.df["path"].apply(fix_path)
        missing = self.df["path"].isnull().sum()
        if missing:
            print(f"Warning: {missing} samples in catalog.csv could not be resolved and will be dropped")
        self.df = self.df[self.df["path"].notnull()].reset_index(drop=True)

        self.transforms = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transforms(img)

        label = int(row["label"])
        video_id = row["video_id"]

        return img, label, video_id
