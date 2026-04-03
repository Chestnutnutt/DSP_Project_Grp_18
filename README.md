# Environmental Sound Classification with DSP + Machine Learning

## Project Overview
This project builds an environmental sound classification system using digital signal processing (DSP) feature extraction and machine learning.  
The goal is to improve robustness for clean, noisy, and bandlimited audio clips by combining preprocessing, handcrafted spectral features, and supervised classification.

## Objectives
- Build a reproducible DSP pipeline for audio preprocessing and feature extraction.
- Improve classification performance over the baseline model.
- Evaluate robustness on degraded audio conditions such as noise and bandlimiting.
- Document the system clearly enough for another student to rerun all experiments.
---

## Repository Structure
```text
project-root/
├── README.md
├── requirements.txt
├── notebooks/
│   └── environmental_sound_classification.ipynb
├── models/
│   └── final_model.joblib
├── results/
│   ├── metrics.csv
└── data/
    ├── train/
    └── submission/
```

### File Roles
- `preprocessing.py`: audio loading, trimming, normalization, bandlimited detection, enhancement.

---

## DSP Implementation
### 1. Audio Preprocessing
The preprocessing stage standardizes audio before feature extraction. The pipeline includes:
- Mono loading and resampling to 16 kHz
- Amplitude normalization
- Silence trimming
- Padding/truncation to a fixed duration
- Bandlimited-audio detection using spectral cutoff analysis
- Near-cutoff enhancement for detected bandlimited clips

### 2. Feature Extraction
The system uses handcrafted DSP features to capture temporal and spectral characteristics of environmental sounds:
- MFCC statistics
- Delta and delta-delta MFCC features
- Log-mel spectrogram statistics
- Spectral centroid
- Spectral bandwidth
- Zero-crossing rate
- Amplitude envelope statistics


### 3. DSP Design Rationale

These features were selected because environmental sounds differ strongly in timbre, spectral distribution, and transient structure.  
For example, percussive sounds such as knocking and gunshots tend to have sharp temporal changes, while sounds like rain or engine noise exhibit more sustained spectral energy patterns.

---

## Model and Experiments
### Model
The main classifier used in this project is:
- `Logistic Regression` / `SVM` / `Random Forest` / `XGBoost`  

### Experimental Setup
Experiments were conducted to compare:
- Baseline features vs improved DSP features
- With and without preprocessing
- With and without bandlimited enhancement
- Different classifier choices and hyperparameters

### Evaluation Metric
The primary evaluation metric is:
- Macro F1-score

Additional metrics:
- Accuracy
- Per-class precision/recall/F1
- Confusion matrix

### Final Model Choice
The final model was selected based on validation Macro F1-score and robustness across degraded audio conditions.  
Briefly explain why it was chosen over the alternatives.

---

## Results and Discussion
### Key Findings
- Improved preprocessing increased robustness for degraded audio.
- Adding richer spectral descriptors improved class separation.
- Bandlimited detection helped identify clips affected by high-frequency loss.
- Some classes remained difficult due to overlapping acoustic characteristics.

### Error Analysis
Common confusion cases included:
- `class A` vs `class B`
- `class C` vs `class D`

Possible reasons:
- Similar spectral profiles
- Background noise contamination
- Short-duration transient events
- Loss of high-frequency content in bandlimited samples

---

## Reproducibility
### Environment Setup


### Data Requirements


### Outputs


### Reproducibility Notes
- Random seed is fixed where applicable.
- Audio is resampled to a consistent sampling rate.
- Feature extraction uses the same configuration for training and inference.
- The final model and experiment settings are recorded in the repository.

---

## Limitations
- Handcrafted DSP features may not capture all complex sound patterns.
- Some classes are difficult to separate with classical ML models alone.
- Performance may be sensitive to heavy distortion or unseen noise conditions.
