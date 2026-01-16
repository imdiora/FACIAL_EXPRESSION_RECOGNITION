
# Facial Expression Recognition with DDAMFN  
**Dual Dynamic Attention Mixed Feature Network**

A PyTorch-based project for **facial emotion recognition**, combining **mixed feature representations** with **attention mechanisms** and supporting **real-time webcam inference**.

---

## ✨ Highlights
- Custom attention-based CNN (**DDAMFN**)
- Clean, modular PyTorch code
- Real-time facial expression recognition
- Trained and evaluated on **RAF-DB** and **FER+**
- Pretrained models hosted on **Hugging Face**

---

## 🎥 Demo

**Video-based**

![DDAMFN Demo](demo.gif)

**Sample prediction**

<p align="left">
  <img src="demo.png" width="380" />
</p>

---

## 🧠 Model
DDAMFN integrates:
- mixed low- and high-level facial features  
- dual dynamic attention blocks  
- lightweight design suitable for real-time use  

Core architecture is implemented in `networks/`.

---

## 📂 Project Structure
```

FACIAL_EXPRESSION_RECOGNITION/
├── networks/              # Model & attention modules
├── facial_recognition.py  # Image / video inference
├── real_time.py           # Webcam demo
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup

```bash
git clone https://github.com/imdiora/FACIAL_EXPRESSION_RECOGNITION.git
cd FACIAL_EXPRESSION_RECOGNITION
pip install -r requirements.txt
````

---

## ▶️ Usage

Run real-time demo:

```bash
python real_time.py
```

Run inference on images / video:

```bash
python facial_recognition.py
```

---

## 📦 Pretrained Models

Pretrained DDAMFN weights are available on **Hugging Face**:

👉 [https://huggingface.co/imdiora/ddamfn-facial-expression-recognition](https://huggingface.co/imdiora/ddamfn-facial-expression-recognition)

After downloading:

```
pretrained/
 └── ddamfn_rafdb_best.pth
```

---

## 🚀 Applications

* Facial emotion recognition
* Human–computer interaction
* Affective computing
* Emotion-aware systems

---

## 🛠️ Tech Stack

Python · PyTorch · OpenCV · NumPy

---

## 👩‍💻 Author

**Diyora Bobokulova**
AI Automation & Computer Vision

GitHub: [https://github.com/imdiora](https://github.com/imdiora)
Hugging Face: [https://huggingface.co/imdiora](https://huggingface.co/imdiora)

**Acknowledgement:**
Thanks to **Jamshid Ganiev** for major support with backend logic and debugging.
