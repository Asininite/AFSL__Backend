import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2
from mtcnn import MTCNN
from tqdm import tqdm

from config import FRAMES_DIR, FACES_DIR, IMAGE_SIZE


detector = MTCNN()

def process_frame(frame_path: Path, out_dir: Path):
    img = cv2.imread(str(frame_path))
    if img is None:
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        detections = detector.detect_faces(rgb)
    except Exception:
        # MTCNN occasionally crashes on bad frames
        return

    if not detections:
        return

    # pick largest face
    largest = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = largest["box"]

    x, y = max(0, x), max(0, y)
    face = rgb[y:y + h, x:x + w]

    if face.size == 0:
        return

    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / frame_path.name
    cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))



def process_video(video_dir: Path, out_dir: Path):
    if out_dir.exists():
        return  # skip already processed videos
    
    frames = list(video_dir.glob("*.jpg"))
    for frame in frames:
        process_frame(frame, out_dir)


def main():
    for manipulation in ["real", "deepfakes"]:
        src_root = FRAMES_DIR / manipulation
        dst_root = FACES_DIR / manipulation

        if not src_root.exists():
            print(f"[WARN] Missing {src_root}")
            continue

        videos = list(src_root.iterdir())
        for video in tqdm(videos, desc=f"Processing {manipulation}"):
            process_video(video, dst_root / video.name)


if __name__ == "__main__":
    main()
