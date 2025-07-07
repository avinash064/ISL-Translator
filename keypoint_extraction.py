# modules/keypoints/mediapipe_extractor.py

import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import glob

# MediaPipe models
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

def extract_keypoints_from_frame(frame, hands_model, pose_model, face_model):
    keypoints = []
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Extract using MediaPipe
    hand_result = hands_model.process(image)
    pose_result = pose_model.process(image)
    face_result = face_model.process(image)

    # Hand Landmarks (2 hands)
    for hand_landmarks in [hand_result.left_hand_landmarks, hand_result.right_hand_landmarks]:
        if hand_landmarks:
            keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            keypoints.extend([[0, 0, 0]] * 21)

    # Pose Landmarks
    if pose_result.pose_landmarks:
        keypoints.extend([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark])
    else:
        keypoints.extend([[0, 0, 0]] * 33)

    # Face Mesh (First 100 landmarks)
    if face_result.multi_face_landmarks:
        keypoints.extend([[lm.x, lm.y, lm.z] for lm in face_result.multi_face_landmarks[0].landmark[:100]])
    else:
        keypoints.extend([[0, 0, 0]] * 100)

    return np.array(keypoints).flatten()  # shape: (N,)

def extract_video_keypoints(video_path, fps_sample=5):
    cap = cv2.VideoCapture(video_path)
    keypoint_sequence = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
         mp_pose.Pose(static_image_mode=False) as pose, \
         mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face:

        frame_count = 0
        success = True
        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_count % fps_sample == 0:
                try:
                    kps = extract_keypoints_from_frame(frame, hands, pose, face)
                    keypoint_sequence.append(kps)
                except:
                    continue
            frame_count += 1

    cap.release()
    return np.array(keypoint_sequence)  # shape: (T, D)

def process_dataset(video_dir, out_dir, fps_sample=5):
    os.makedirs(out_dir, exist_ok=True)

    # Recursively find all .mp4 files
    video_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)

    for video_path in tqdm(video_files, desc="Extracting keypoints"):
        # Flatten file path to avoid collisions
        rel_path = os.path.relpath(video_path, video_dir)
        flat_name = rel_path.replace("/", "_").replace(".mp4", ".npy")
        output_path = os.path.join(out_dir, flat_name)

        kps = extract_video_keypoints(video_path, fps_sample=fps_sample)
        if kps.size > 0:
            np.save(output_path, kps)

    print(f"\n✅ Keypoints saved to: {out_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/data/UG/Avinash/MedicalQ&A/SignLanguage/data/isl_datasets",
        help="Path to root folder containing ISL videos"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/UG/Avinash/MedicalQ&A/SignLanguage/data/keypoints/ISL-CSLTR",
        help="Output directory for .npy keypoint sequences"
    )
    parser.add_argument(
        "--fps_sample",
        type=int,
        default=5,
        help="Frame sampling rate (every N frames)"
    )

    args = parser.parse_args()
    process_dataset(args.video_dir, args.out_dir, fps_sample=args.fps_sample)
#     print(f"Error downloading Kaggle datasets: {e}")