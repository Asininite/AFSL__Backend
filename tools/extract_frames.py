import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2
from tqdm import tqdm
from config import FFPP_C23_DIR, FRAMES_DIR

# FF++ structure
REAL_DIR = FFPP_C23_DIR / "original"
FAKE_DIRS = {
    "deepfakes": FFPP_C23_DIR / "Deepfakes",
}

FRAME_STRIDE = 10  # extract 1 frame every 10 frames


def extract_video(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE == 0:
            out_path = out_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()


def process_folder(folder: Path, label: str):
    videos = list(folder.glob("*.mp4"))
    for video in tqdm(videos, desc=f"Extracting {label}"):
        video_id = video.stem
        extract_video(video, FRAMES_DIR / label / video_id)


def main():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Real videos
    process_folder(REAL_DIR, "real")

    # Fake videos
    for name, path in FAKE_DIRS.items():
        process_folder(path, name)


if __name__ == "__main__":
    main()
