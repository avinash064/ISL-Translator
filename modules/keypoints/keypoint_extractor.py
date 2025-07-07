# import cv2
# import os
# import numpy as np
# from tqdm import tqdm
# import mediapipe as mp
# import glob

# # Initialize MediaPipe modules
# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_face = mp.solutions.face_mesh


# def extract_keypoints_from_frame(frame, hands_model, pose_model, face_model):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False

#     hand_result = hands_model.process(image)
#     pose_result = pose_model.process(image)
#     face_result = face_model.process(image)

#     keypoints = []

#     # Hand Landmarks (left & right)
#     for hand_landmarks in [hand_result.left_hand_landmarks, hand_result.right_hand_landmarks]:
#         if hand_landmarks:
#             keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
#         else:
#             keypoints.extend([[0, 0, 0]] * 21)

#     # Pose Landmarks (33)
#     if pose_result.pose_landmarks:
#         keypoints.extend([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark])
#     else:
#         keypoints.extend([[0, 0, 0]] * 33)

#     # Face Mesh (first 100 points)
#     if face_result.multi_face_landmarks:
#         keypoints.extend([[lm.x, lm.y, lm.z] for lm in face_result.multi_face_landmarks[0].landmark[:100]])
#     else:
#         keypoints.extend([[0, 0, 0]] * 100)

#     return np.array(keypoints).flatten()


# def extract_video_keypoints(video_path, fps_sample=5):
#     cap = cv2.VideoCapture(video_path)
#     keypoint_sequence = []

#     with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
#          mp_pose.Pose(static_image_mode=False) as pose, \
#          mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face:

#         frame_idx = 0
#         success = True
#         while success:
#             success, frame = cap.read()
#             if not success:
#                 break

#             if frame_idx % fps_sample == 0:
#                 try:
#                     keypoints = extract_keypoints_from_frame(frame, hands, pose, face)
#                     keypoint_sequence.append(keypoints)
#                 except Exception as e:
#                     print(f"⚠️ Error on frame {frame_idx} in {video_path}: {e}")
#             frame_idx += 1

#     cap.release()
#     return np.array(keypoint_sequence)  # (T, D)


# def process_dataset(video_dir, out_dir, fps_sample=5, overwrite=False):
#     os.makedirs(out_dir, exist_ok=True)
#     video_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)

#     for video_path in tqdm(video_files, desc="Extracting keypoints"):
#         rel_path = os.path.relpath(video_path, video_dir)
#         flat_name = rel_path.replace(os.sep, "_").replace(".mp4", ".npy")
#         output_path = os.path.join(out_dir, flat_name)

#         if os.path.exists(output_path) and not overwrite:
#             continue  # Skip if already extracted

#         keypoints = extract_video_keypoints(video_path, fps_sample)
#         if keypoints.size > 0:
#             np.save(output_path, keypoints)

#     print(f"\n✅ Keypoint extraction complete. Saved to: {out_dir}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Extract keypoints from ISL video dataset")
#     parser.add_argument(
#         "--video_dir",
#         type=str,
#         required=True,
#         help="Path to root directory containing ISL videos"
#     )
#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         required=True,
#         help="Directory to save extracted .npy keypoints"
#     )
#     parser.add_argument(
#         "--fps_sample",
#         type=int,
#         default=5,
#         help="Extract keypoints every Nth frame (default: 5)"
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Re-process videos even if output already exists"
#     )

#     args = parser.parse_args()
#     process_dataset(
#         args.video_dir,
#         args.out_dir,
#         fps_sample=args.fps_sample,
#         overwrite=args.overwrite
#     )
import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import glob

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


def extract_keypoints_from_frame(frame, hands_model, pose_model, face_model):
    keypoints = []
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run MediaPipe models
    hand_result = hands_model.process(image)
    pose_result = pose_model.process(image)
    face_result = face_model.process(image)

    # --- Hand Keypoints (21 each) ---
    left_hand = [[0, 0, 0]] * 21
    right_hand = [[0, 0, 0]] * 21

    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        for i, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            label = hand_result.multi_handedness[i].classification[0].label
            points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            if label == "Left":
                left_hand = points
            elif label == "Right":
                right_hand = points

    keypoints.extend(left_hand)
    keypoints.extend(right_hand)

    # --- Pose Keypoints (33) ---
    if pose_result.pose_landmarks:
        keypoints.extend([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark])
    else:
        keypoints.extend([[0, 0, 0]] * 33)

    # --- Face Keypoints (First 100) ---
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
                except Exception as e:
                    print(f"⚠️ Error on frame {frame_count} in {video_path}: {e}")
            frame_count += 1

    cap.release()
    return np.array(keypoint_sequence)  # shape: (T, D)


def process_dataset(video_dir, out_dir, fps_sample=5, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)

    # Recursively find all .mp4 files
    video_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {video_dir}")

    for video_path in tqdm(video_files, desc="Extracting keypoints"):
        rel_path = os.path.relpath(video_path, video_dir)
        flat_name = rel_path.replace("/", "_").replace(".mp4", ".npy")
        output_path = os.path.join(out_dir, flat_name)

        if os.path.exists(output_path) and not overwrite:
            continue

        try:
            kps = extract_video_keypoints(video_path, fps_sample=fps_sample)
            if kps.size > 0:
                np.save(output_path, kps)
        except Exception as e:
            print(f"❌ Failed processing {video_path}: {e}")

    print(f"\n✅ Keypoints saved to: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to root folder containing ISL videos"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for .npy keypoint sequences"
    )
    parser.add_argument(
        "--fps_sample",
        type=int,
        default=5,
        help="Sample every Nth frame"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files"
    )

    args = parser.parse_args()
    process_dataset(
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        fps_sample=args.fps_sample,
        overwrite=args.overwrite
    )
# This script extracts keypoints from ISL videos using MediaPipe and saves them as .npy files.