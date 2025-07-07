# 🤟 ISL Translator - Real-Time Indian Sign Language to Speech/Text System

> A real-time deep learning-based system for translating **Indian Sign Language (ISL)** gestures into spoken text using **MediaPipe**, **PyTorch**, and **TTS** engines like Coqui-TTS and pyttsx3. Designed to enhance communication accessibility and bridge the gap between signers and non-signers.

---

## 🚀 Key Features

- 🎥 **Real-time Keypoint Extraction** using **MediaPipe** (Hands + Pose + Face)
- 🤖 **Deep Learning Model** for gesture-to-text mapping
- 🔊 **Text-to-Speech Integration** for verbal output
- 🖥️ Supports **both Webcam and Video File** inputs
- 🔁 Smoothing and confidence filtering for stable prediction
- 🧠 Multi-task loss design for better gesture understanding
- 💡 Modular, production-ready pipeline with clear APIs
- 🌐 Designed for research, healthcare, education, and accessibility

---

## 🎯 Real-World Applications

| Sector        | Use Case                                                                 |
|---------------|--------------------------------------------------------------------------|
| 🏥 Healthcare | Doctor-patient communication in silent wards or speech-impaired cases   |
| 🎓 Education  | Teaching and learning sign language via feedback and automation         |
| 🧏 Accessibility | Public service counters, train stations, airports for ISL speakers     |
| 🛒 Retail      | Gesture-based customer service kiosks or checkout counters              |
| 🏠 Smart Homes | Hands-free control using sign commands                                 |
| 🔬 Research    | Building datasets, evaluating sign language learning, HCI studies       |

---

## 📦 Installation

### 1. Clone the Repo
```bash
git clone git@github.com:avinash064/ISL-Translator.git
cd ISL-Translator
```

### 2. Create Environment
```bash
conda create -n isl-translator python=3.10 -y
conda activate isl-translator
pip install -r requirements.txt
```

> 💡 Optionally, install Coqui-TTS:
```bash
pip install TTS
```

---

## 🧠 Model Architecture

- 🔹 **Input**: 3D Keypoints (Pose + Hand + Face) → (T, D) shape
- 🔹 **Encoder**: BiLSTM/Transformer
- 🔹 **Classifier**: Fully Connected + Softmax
- 🔹 **Loss**: Custom Multi-task Loss combining:
  - CrossEntropyLoss
  - CTC Loss
  - Attention Alignment (optional)
  - Label Smoothing

---

## 🧪 Training Pipeline

```bash
python -m models.Training \
  --keypoint_dir "/data/.../keypoints/ISL-CSLTR" \
  --batch_size 32 \
  --epochs 20 \
  --checkpoint "checkpoints/model.pt"
```

Key modules:
- `keypoint_extractor.py`: Recursively extract `.npy` keypoints from videos
- `loss.py`: Multi-loss integration with flexibility
- `training_validation.py`: Robust training script with checkpointing and logging

---

## 💻 Real-Time Inference

```bash
python -m inference.realtime_pipeline \
  --model_ckpt checkpoints/model.pt \
  --backend torch \
  --fps 15 \
  --smooth 5 \
  --tts en_XX
```

### ✅ Features
- Supports both **live webcam** and **video input** (`--video`)
- Smooths predictions over windowed majority voting
- Converts output to **speech using pyttsx3 or Coqui-TTS**

---

## 🛠 Project Structure

```
ISL-Translator/
│
├── modules/                    # Key reusable modules
│   ├── keypoint_extractor.py
│   ├── keypoint_encoder.py
│   ├── model.py
│   └── text_decoder.py
│
├── datasets/
│   └── keypoint_loader.py
│
├── models/
│   ├── Training.py
│   ├── loss.py
│   └── training_validation.py
│
├── inference/
│   └── realtime_pipeline.py
│
├── data/                       # Keypoint .npy files and dataset structure
├── checkpoints/
├── README.md
└── requirements.txt
```

---

## 🔭 Future Work

- [ ] Integrate **NeRF/Gaussian Splatting** for 3D hand mesh
- [ ] Add **language modeling (BERT/LLM)** for better output fluency
- [ ] Deploy on **Jetson Nano / Raspberry Pi**
- [ ] Build web app with **Streamlit/FastAPI**
- [ ] Support **bi-directional translation (text → ISL)**

---

## 🤝 Contributions

Contributions, feedback, and ideas are welcome!  
Please raise issues or open PRs.

---

## 🧾 License

MIT License © 2025 [Avinash Kashyap](https://github.com/avinash064)

---

> _"Empowering communication, one sign at a time."_ ✨
