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

## Core preprocessing

| Step | What the code does | Purpose |
|---|---|---|
| Audio loading | Loads each waveform with `librosa.load(path, sr=16000, mono=True)` and converts it to `float32` | Standardizes sampling rate and channel format so all clips are processed consistently |
| Invalid-value cleanup | Replaces NaN values with valid numeric values using `np.nan_to_num(...)` | Prevents corrupted samples from breaking later preprocessing or feature extraction steps |
| Peak normalization | Scales the waveform so the maximum absolute amplitude becomes 1 using `peak_normalise(y)` | Makes threshold-based operations more stable across clips with different loudness levels |
| Silence trimming | Removes leading and trailing low-energy regions with `librosa.effects.trim(y, top_db=30)` | Reduces unnecessary silence so features focus more on the actual sound event |
| Empty-signal fallback | If trimming removes everything, the function returns a 1-second zero signal instead of an empty array | Ensures downstream feature extraction always receives a valid waveform |

## Adaptive preprocessing

This table describes the conditional logic that is only applied when the clip appears noisy or band-limited.

| Condition | What the code checks | Action taken | Why it helps |
|---|---|---|---|
| Low-SNR audio | The preprocessing function estimates SNR from frame-energy percentiles and compares it against `denoise_if_snr_below`; in your feature extraction call, this threshold is set to 5.0 dB | If the estimated SNR is below the threshold, the waveform is processed with `librosa.effects.preemphasis(y, coef=0.95)` | This emphasizes higher-frequency content and can improve robustness for degraded recordings |
| Band-limited audio | The code estimates an effective cutoff frequency from the STFT, then marks audio as band-limited when the cutoff is at or below 5500 Hz and the spectral drop is at least 18 dB | If detected, `enhance_near_cutoff(...)` applies a band-pass boost near the estimated cutoff, with a default boost of 20 dB and peak re-normalization afterward | This tries to recover useful energy near the cutoff region for clips that have restricted high-frequency content |

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
