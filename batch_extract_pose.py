import os
import csv
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BAGS_DIR   = r"C:\bag\Validacion"
OUT_DIR    = r"C:\bag\poses_validacion"
MODEL_PATH = r"C:\bag\Extraccion de pose\pose_landmarker.task"
FPS = 30
GLOBAL_TS_MS = 0


os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.exists(MODEL_PATH), "No existe el modelo .task"
assert os.path.isdir(BAGS_DIR), "No existe la carpeta de bags"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    # más permisivo para no perder tantos frames:
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

def process_one_bag(bag_path: str, out_csv: str) -> tuple[int, int]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    profile = pipeline.start(config)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intr = depth_stream.as_video_stream_profile().get_intrinsics()

    rows = []
    frame_idx = 0
    global GLOBAL_TS_MS


    try:
        while True:
            try:
                frameset = pipeline.wait_for_frames(15000)
            except RuntimeError:
                break

            frameset = align.process(frameset)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = GLOBAL_TS_MS
            GLOBAL_TS_MS += int(1000 / FPS)  # 33 ms aprox por frame

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lmks = result.pose_landmarks[0]
                for j, lm in enumerate(lmks):
                    px = int(np.clip(lm.x * w, 0, w - 1))
                    py = int(np.clip(lm.y * h, 0, h - 1))
                    z = depth_frame.get_distance(px, py)
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [px, py], z)
                    rows.append([frame_idx, j, X, Y, Z])

            frame_idx += 1

    finally:
        pipeline.stop()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "joint", "X", "Y", "Z"])
        w.writerows(rows)

    return frame_idx, len(rows)

bags = [f for f in os.listdir(BAGS_DIR) if f.lower().endswith(".bag")]
bags.sort()

print("Bags encontrados:", len(bags))

for i, fname in enumerate(bags, 1):
    bag_path = os.path.join(BAGS_DIR, fname)
    out_csv = os.path.join(OUT_DIR, os.path.splitext(fname)[0] + "_pose.csv")
    try:
        frames, pts = process_one_bag(bag_path, out_csv)
        print(f"[{i}/{len(bags)}] {fname} -> frames={frames}, points={pts}")
    except Exception as e:
        print(f"[{i}/{len(bags)}] ERROR en {fname}: {e}")
        continue
    
landmarker.close()
print("LISTO. CSVs en:", OUT_DIR)
