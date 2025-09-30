# Chest X-Ray CNN Pneumonia Detection — Custom CNN with ResNet Benchmark

This project explores **pneumonia detection from chest X-ray images using deep learning**, with the primary goal of **designing and training a custom CNN from scratch**.

After developing and tuning my own architecture, I later introduced **ResNet18 (transfer learning)** as a **sanity check / performance benchmark** — not as the main focus, but to validate how well my custom model stacks up against a well-established baseline.

**by Keela Ta**

---

## 🚀 Project Overview

- ✅ **Phase 1 — Build my own CNN from first principles**
  - Designed the full architecture manually
  - Iteratively tuned augmentation, dropout, normalization, and learning rate scheduling
  - Reached **97% validation accuracy / 77% test accuracy**
- 🔍 **Phase 2 — Benchmark against ResNet18**
  - Used only to **verify whether the custom model is under/over-performing**
  - Result: **ResNet hit ~83% test accuracy** — confirming that **my CNN performs competitively**, though with different bias characteristics

This was a **self-driven research project**, motivated by an interest in **medical AI applications** and hands-on **model experimentation**.

---

## 📂 Dataset

- **Source:** Kaggle — Chest X-Ray Pneumonia
- **Classes:** `NORMAL` vs `PNEUMONIA`
- **Preprocessing:**
  - Resize → 128×128 (CNN) / 224×224 (ResNet)
  - Random flips & rotations for augmentation
  - Grayscale → 3-channel conversion
  - Normalisation

---

## 🛠️ Model Architectures

### Custom CNN (Primary Model)

- 3× Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
- Fully connected classifier head
- Trained with **Adam + OneCycleLR Scheduler**
- **Strengths:** Lower loss, more cautious predictions  
- **Weaknesses:** Lower recall on *NORMAL* class — tends to predict pneumonia more often

### ResNet18 Benchmark (Secondary)

- Pretrained on ImageNet
- Final FC layer swapped for binary output
- Used to **cross-check performance ceiling**

---

## 📊 Results Summary

| Model          | Test Accuracy | Notes                                    |
|----------------|---------------|------------------------------------------|
| **Custom CNN** | **0.77**      | Slightly overfits, cautious predictions  |
| **ResNet18**   | **0.83**      | Stronger generalisation but more volatile|

> **Conclusion:** My custom CNN holds up well — ResNet performs better overall, but the gap is small, validating that the architecture I designed is fundamentally sound.

---

## 🛠️ How to Run

```bash
git clone https://github.com/kt1156/Chest-X-Rays-CNN
cd Chest-X-Rays-CNN
# Upload kaggle.json in Colab and run notebook cells

chest_xray/
 ├── train/
 ├── val/    # If missing, split manually
 └── test/
