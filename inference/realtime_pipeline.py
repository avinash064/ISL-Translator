# #!/usr/bin/env python3
# """
# inference/realtime_pipeline.py – Enhanced real‑time sign‑to‑text translation
# ------------------------------------------------------------------------------
# Key upgrades:
#   • Sliding‑window majority‑vote smoothing (configurable)
#   • Keypoint visibility confidence filter (MediaPipe landmark.visibility)
#   • Optional ONNX/TorchScript backend for ultra‑fast inference
#   • Optional multilingual TTS (Coqui‑TTS fallback to pyttsx3)
#   • Live FPS + latency overlay

# Run example:
# ```bash
# python inference/realtime_pipeline.py \
#   --model_ckpt checkpoints/multitask/model.pt \
#   --backend torch            # or torchscript/onnx \
#   --tokenizer facebook/mbart-large-50-many-to-many-mmt \
#   --max_len 120 \
#   --fps 15 \
#   --smooth 5 \
#   --conf_thresh 0.5 \
#   --tts en_XX
# ```
# """
# import argparse
# import time
# from collections import deque
# from pathlib import Path

# import cv2
# import numpy as np
# import torch
# from transformers import MBart50TokenizerFast
# import mediapipe as mp

# try:
#     import TTS
#     from TTS.utils.manage import ModelManager
# except ImportError:
#     TTS = None
#     print("ℹ️  Coqui‑TTS not installed, falling back to pyttsx3 (if enabled)")

# try:
#     import pyttsx3
# except ImportError:
#     pyttsx3 = None

# from modules.keypoint_encoder import KeypointEncoder
# from models.multilingual_decoder import MultilingualTransformerDecoder


# def speak(text: str, lang: str):
#     if TTS is not None:
#         mgr = ModelManager()
#         m, cfg, _ = mgr.download_model('tts_models/multilingual/multi-dataset/your_tts')
#         synth = TTS.utils.synthesizer.Synthesizer(m, cfg)
#         wav = synth.tts(text, speaker_wav=None, language=lang)
#         TTS.utils.audio.play_audio(wav)
#     elif pyttsx3 is not None:
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()


# def visibility_score(landmarks):
#     return np.mean([lm.visibility for lm in landmarks]) if landmarks else 0.0


# def extract_frame_keypoints(frame, handles, buffer, max_len, fps_step, idx, conf_thresh):
#     if idx % fps_step != 0:
#         return False

#     hands, pose, face = handles
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img.flags.writeable = False
#     hr = hands.process(img)
#     pr = pose.process(img)
#     fr = face.process(img)

#     if pr.pose_landmarks and visibility_score(pr.pose_landmarks.landmark) < conf_thresh:
#         return False

#     kps = []
#     left = [[0,0,0]]*21
#     right = [[0,0,0]]*21
#     if hr.multi_hand_landmarks and hr.multi_handedness:
#         for lm_set, info in zip(hr.multi_hand_landmarks, hr.multi_handedness):
#             if visibility_score(lm_set.landmark) < conf_thresh:
#                 continue
#             pts = [[p.x,p.y,p.z] for p in lm_set.landmark]
#             if info.classification[0].label == 'Left': left = pts
#             else: right = pts
#     kps.extend(left); kps.extend(right)

#     if pr.pose_landmarks:
#         kps.extend([[p.x,p.y,p.z] for p in pr.pose_landmarks.landmark])
#     else:
#         kps.extend([[0,0,0]]*33)

#     if fr.multi_face_landmarks and visibility_score(fr.multi_face_landmarks[0].landmark) >= conf_thresh:
#         kps.extend([[p.x,p.y,p.z] for p in fr.multi_face_landmarks[0].landmark[:100]])
#     else:
#         kps.extend([[0,0,0]]*100)

#     buffer.append(np.array(kps, dtype=np.float32))
#     if len(buffer) > max_len:
#         buffer.popleft()
#     return True


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_ckpt', required=True)
#     parser.add_argument('--tokenizer', default='facebook/mbart-large-50-many-to-many-mmt')
#     parser.add_argument('--backend', choices=['torch','torchscript','onnx'], default='torch')
#     parser.add_argument('--max_len', type=int, default=120)
#     parser.add_argument('--fps', type=int, default=15)
#     parser.add_argument('--smooth', type=int, default=5)
#     parser.add_argument('--conf_thresh', type=float, default=0.5)
#     parser.add_argument('--tts', default=None)
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tok = MBart50TokenizerFast.from_pretrained(args.tokenizer)

#     if args.backend == 'torch':
#         encoder = KeypointEncoder(input_dim=(21+21+33+100)*3, hidden_dim=512).to(device)
#         decoder = MultilingualTransformerDecoder(tok, hidden_dim=512).to(device)
#         ckpt = torch.load(args.model_ckpt, map_location=device)
#         encoder.load_state_dict(ckpt['encoder'])
#         decoder.load_state_dict(ckpt['decoder'])
#     else:
#         enc_path = Path(args.model_ckpt).with_suffix('_enc.ts')
#         dec_path = Path(args.model_ckpt).with_suffix('_dec.ts')
#         encoder = torch.jit.load(str(enc_path)).to(device)
#         decoder = torch.jit.load(str(dec_path)).to(device)

#     encoder.eval(); decoder.eval()

#     mp_h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
#     mp_p = mp.solutions.pose.Pose(static_image_mode=False)
#     mp_f = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
#     handles = (mp_h, mp_p, mp_f)

#     cap = cv2.VideoCapture(0)
#     buffer = deque(maxlen=args.max_len)
#     votes = deque(maxlen=args.smooth)
#     fps_step = max(1, int(30/args.fps))
#     idx = 0
#     last_pred = "..."
#     last_time = time.time()

#     with mp_h, mp_p, mp_f:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             idx += 1
#             extract_frame_keypoints(frame, handles, buffer, args.max_len, fps_step, idx, args.conf_thresh)
#             if len(buffer) == args.max_len:
#                 seq = torch.tensor([list(buffer)], dtype=torch.float32, device=device)
#                 with torch.no_grad():
#                     mem = encoder(seq)
#                     pred = decoder.translate(mem, tgt_lang='en_XX', beam=False)
#                 votes.append(pred)
#                 if len(votes) == args.smooth:
#                     pred = max(set(votes), key=votes.count)
#                 last_pred = pred
#                 if args.tts and (idx % (fps_step*args.smooth) == 0):
#                     speak(pred, args.tts)

#             now = time.time()
#             fps_val = 1.0/(now-last_time)
#             last_time = now
#             cv2.putText(frame, f"FPS:{fps_val:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),1)
#             cv2.putText(frame, last_pred, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

#             cv2.imshow('Sign2Text', frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
# # -----------------------------------------------------------------------------
# # This code is part of the MedicalQ&A project, licensed under the MIT License.
# # See LICENSE for details.
# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# # This code is part of the MedicalQ&A project, licensed under the MIT License.
# # See LICENSE for details.
# import argparse
# import cv2
# import torch
# import numpy as np
# import pyttsx3
# from tqdm import deque
# from pathlib import Path
# from modules.model import SignLanguageModel  # Replace with actual model import
# from modules.keypoint_extractor import extract_keypoints_from_frame
# from modules.text_decoder import decode_output  # Replace with actual text decoding logic


# def speak_text(text, tts_engine):
#     if text:
#         tts_engine.say(text)
#         tts_engine.runAndWait()


# def smooth_predictions(pred_queue, window=5):
#     # Majority voting
#     if len(pred_queue) < window:
#         return ""
#     return max(set(pred_queue), key=pred_queue.count)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_ckpt', type=str, required=True, help='Path to model checkpoint')
#     parser.add_argument('--video', type=str, help='Path to input video file (optional)')
#     parser.add_argument('--backend', type=str, default='torch', choices=['torch'], help='Inference backend')
#     parser.add_argument('--fps', type=int, default=15, help='Frames per second to sample')
#     parser.add_argument('--smooth', type=int, default=5, help='Prediction smoothing window')
#     parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold')
#     parser.add_argument('--tts', type=str, default='en', help='TTS language')
#     args = parser.parse_args()

#     print("\n🔄 Loading model...")
#     model = SignLanguageModel()
#     model.load_state_dict(torch.load(args.model_ckpt, map_location='cpu'))
#     model.eval()

#     # Text-to-speech engine
#     tts_engine = pyttsx3.init()
#     tts_engine.setProperty('rate', 150)

#     # Webcam or video file
#     video_source = args.video if args.video else 0
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print(f"❌ Error: Could not open video source: {video_source}")
#         return

#     pred_queue = deque(maxlen=args.smooth)
#     frame_count = 0

#     with torch.no_grad():
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_count % args.fps == 0:
#                 try:
#                     keypoints = extract_keypoints_from_frame(frame)
#                     keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)

#                     logits = model(keypoints_tensor)
#                     pred_idx = torch.argmax(logits, dim=-1).item()
#                     pred_text = decode_output(pred_idx)

#                     pred_queue.append(pred_text)
#                     smoothed = smooth_predictions(pred_queue, window=args.smooth)

#                     # Overlay text on frame
#                     cv2.putText(frame, f"{smoothed}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

#                     print(f"🧠 Predicted: {smoothed}")
#                     speak_text(smoothed, tts_engine)

#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")

#             cv2.imshow("ISL Translator", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
inference/realtime_pipeline.py – Enhanced real‑time sign‑to‑text translation
------------------------------------------------------------------------------
Key upgrades:
  • Sliding‑window majority‑vote smoothing (configurable)
  • Keypoint visibility confidence filter (MediaPipe landmark.visibility)
  • Optional ONNX/TorchScript backend for ultra‑fast inference
  • Optional multilingual TTS (Coqui‑TTS fallback to pyttsx3)
  • Live FPS + latency overlay

Run example:
```bash
python inference/realtime_pipeline.py \
  --model_ckpt checkpoints/multitask/model.pt \
  --backend torch            # or torchscript/onnx \
  --tokenizer facebook/mbart-large-50-many-to-many-mmt \
  --max_len 120 \
  --fps 15 \
  --smooth 5 \
  --conf_thresh 0.5 \
  --tts en_XX
```
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import MBart50TokenizerFast
import mediapipe as mp

try:
    import TTS
    from TTS.utils.manage import ModelManager
except ImportError:
    TTS = None
    print("ℹ️  Coqui‑TTS not installed, falling back to pyttsx3 (if enabled)")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from modules.keypoint_encoder import KeypointEncoder
from models.multilingual_decoder import MultilingualTransformerDecoder


def speak(text: str, lang: str):
    if TTS is not None:
        mgr = ModelManager()
        m, cfg, _ = mgr.download_model('tts_models/multilingual/multi-dataset/your_tts')
        synth = TTS.utils.synthesizer.Synthesizer(m, cfg)
        wav = synth.tts(text, speaker_wav=None, language=lang)
        TTS.utils.audio.play_audio(wav)
    elif pyttsx3 is not None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def visibility_score(landmarks):
    return np.mean([lm.visibility for lm in landmarks]) if landmarks else 0.0


def extract_frame_keypoints(frame, handles, buffer, max_len, fps_step, idx, conf_thresh):
    if idx % fps_step != 0:
        return False

    hands, pose, face = handles
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    hr = hands.process(img)
    pr = pose.process(img)
    fr = face.process(img)

    if pr.pose_landmarks and visibility_score(pr.pose_landmarks.landmark) < conf_thresh:
        return False

    kps = []
    left = [[0,0,0]]*21
    right = [[0,0,0]]*21
    if hr.multi_hand_landmarks and hr.multi_handedness:
        for lm_set, info in zip(hr.multi_hand_landmarks, hr.multi_handedness):
            if visibility_score(lm_set.landmark) < conf_thresh:
                continue
            pts = [[p.x,p.y,p.z] for p in lm_set.landmark]
            if info.classification[0].label == 'Left': left = pts
            else: right = pts
    kps.extend(left); kps.extend(right)

    if pr.pose_landmarks:
        kps.extend([[p.x,p.y,p.z] for p in pr.pose_landmarks.landmark])
    else:
        kps.extend([[0,0,0]]*33)

    if fr.multi_face_landmarks and visibility_score(fr.multi_face_landmarks[0].landmark) >= conf_thresh:
        kps.extend([[p.x,p.y,p.z] for p in fr.multi_face_landmarks[0].landmark[:100]])
    else:
        kps.extend([[0,0,0]]*100)

    buffer.append(np.array(kps, dtype=np.float32))
    if len(buffer) > max_len:
        buffer.popleft()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', required=True)
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50-many-to-many-mmt')
    parser.add_argument('--backend', choices=['torch','torchscript','onnx'], default='torch')
    parser.add_argument('--max_len', type=int, default=120)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--smooth', type=int, default=5)
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--tts', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = MBart50TokenizerFast.from_pretrained(args.tokenizer)

    if args.backend == 'torch':
        encoder = KeypointEncoder(input_dim=(21+21+33+100)*3, hidden_dim=512).to(device)
        decoder = MultilingualTransformerDecoder(tok, hidden_dim=512).to(device)
        ckpt = torch.load(args.model_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        enc_path = Path(args.model_ckpt).with_suffix('_enc.ts')
        dec_path = Path(args.model_ckpt).with_suffix('_dec.ts')
        encoder = torch.jit.load(str(enc_path)).to(device)
        decoder = torch.jit.load(str(dec_path)).to(device)

    encoder.eval(); decoder.eval()

    mp_h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    mp_p = mp.solutions.pose.Pose(static_image_mode=False)
    mp_f = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    handles = (mp_h, mp_p, mp_f)

    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=args.max_len)
    votes = deque(maxlen=args.smooth)
    fps_step = max(1, int(30/args.fps))
    idx = 0
    last_pred = "..."
    last_time = time.time()

    with mp_h, mp_p, mp_f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            extract_frame_keypoints(frame, handles, buffer, args.max_len, fps_step, idx, args.conf_thresh)
            if len(buffer) == args.max_len:
                seq = torch.tensor([list(buffer)], dtype=torch.float32, device=device)
                with torch.no_grad():
                    mem = encoder(seq)
                    pred = decoder.translate(mem, tgt_lang='en_XX', beam=False)
                votes.append(pred)
                if len(votes) == args.smooth:
                    pred = max(set(votes), key=votes.count)
                last_pred = pred
                if args.tts and (idx % (fps_step*args.smooth) == 0):
                    speak(pred, args.tts)

            now = time.time()
            fps_val = 1.0/(now-last_time)
            last_time = now
            cv2.putText(frame, f"FPS:{fps_val:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),1)
            cv2.putText(frame, last_pred, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            cv2.imshow('Sign2Text', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
