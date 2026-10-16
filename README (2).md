# Envisioned Speech Recognition Framework Using Spectral Graph Wavelet Transform on Multi-Channel EEG Signals

**Project Repository:** [EEG_Envisioned_Speech1](https://github.com/karthikeya-18-05-07/EEG_Envisioned_Speech1)

## Project Team

- **S. Karthikeya** — 2520040004
- **Shanmukh Pranay** — 2520040070
- **Laxmi Ganesh** — 2520040109

**Project Guide:** V. Sahiti  
**Department:** Electronics and Communication Engineering  
**Institution:** KL University, Hyderabad  
**Academic Year:** 2025–26

## Project Overview

This project explores imagined-speech recognition from multi-channel EEG signals. It uses the EEGIS dataset and a machine-learning pipeline to classify nine classes: Rest and eight words—Ayuda (Help), Baño (Bath), Dolor (Pain), Gracias (Thank you), Hambre (Hunger), No, Sed (Thirst), and Sí (Yes).

The goal is to investigate whether EEG preprocessing, multi-band feature extraction, graph-based smoothing, and machine-learning classification can support a brain–computer interface for communication assistance.

## Methodology

1. **EEGIS data ingestion:** Reads the six representations/bands—raw, delta, theta, alpha, beta, and gamma—and arranges valid trials into a multi-channel tensor.
2. **Euclidean Alignment (EA):** Applies subject-wise spatial alignment to reduce differences in EEG covariance.
3. **Feature extraction:** Extracts differential entropy, spatial covariance, and band-ratio features, for a total of 756 features per trial.
4. **Graph construction and smoothing:** Builds a 14-channel adjacency matrix from absolute channel correlations, computes the graph Laplacian, and applies first-order graph smoothing to differential-entropy features.
5. **Feature selection:** Uses SelectKBest with ANOVA scoring to select 180 features.
6. **Classification:** Uses an ensemble combining an RBF Support Vector Machine (SVM) and Extra Trees with soft voting.
7. **Neural gateway:** Suppresses the decoded output if the predicted class is Rest or the model confidence is below 40%; otherwise, it displays the predicted word.
8. **Evaluation and demonstration:** Includes stratified five-fold cross-validation, subject-dependent evaluation, and a Streamlit interface for inspecting predictions and EEG signals.

## Preliminary Results

The Review 2 experiment reported the following results for the evaluated run:

| Metric | Reported result |
|---|---:|
| Global mean accuracy (stratified five-fold CV) | 78.66% |
| Global mean F1-score | 78.70% |
| Mean subject-dependent accuracy | 81.36% |
| Random-chance reference for nine classes | 11.11% |

These are preliminary results from the reported evaluation. They should not be interpreted as proof of generalisation to unseen subjects.

## Implementation Status and Limitations

The current implementation uses first-order graph diffusion/smoothing. The full 50th-degree Chebyshev SGWT kernel described in the project plan is **not yet implemented**.

The following work remains pending or proposed:

- Leave-One-Subject-Out (LOSO) cross-validation.
- Ablation studies for EA, graph smoothing, individual classifiers, and EEG bands.
- Evaluation of alternative graph constructions, including PLV connectivity.
- Sweeps of the graph-smoothing parameter and gateway confidence threshold.
- Runtime measurement and final validation.

The reported subject IDs are assigned by chunk order rather than verified dataset metadata. Stratified folds may mix subjects and neighbouring chunks, so the reported accuracy may be optimistic. Graph and alignment calculations also use all trials in the current evaluation procedure; strict held-out evaluation remains necessary.

## Software Requirements

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Joblib
- scikit-learn

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

If the installed scikit-learn version is incompatible with the saved model, use the version specified in `requirements.txt` and a compatible Python environment.

## Run the Application

1. Clone or download the repository.
2. Open the project folder in VS Code.
3. Install the dependencies.
4. Ensure the required model and data files are available in the `eeg_app/` directory.
5. Start the Streamlit app:

```bash
streamlit run app.py
```

6. Open the local URL printed by Streamlit in your browser.

## Required Local Model and Data Files

The application may require the following files in `eeg_app/`:

- `model.pkl`
- `test_features.npy`
- `test_labels.npy`
- `vocab.pkl`
- `raw_trials.npy`

These files are not guaranteed to be included in the GitHub repository. If they are excluded due to size, dataset licensing, or privacy considerations, obtain them from the project team and place them in the expected directory before running the app.

## Repository

Source code and project updates:  
https://github.com/karthikeya-18-05-07/EEG_Envisioned_Speech1

## References

1. D. Pachori et al., “EEG-based envisioned speech recognition system using spectral graph wavelet transform,” 2024.
2. P. Tirupattur et al., “ThoughtViz,” ACM Multimedia, 2018.
3. P. Kumar and E. Scheme, “A deep spatio-temporal model for EEG-based imagined speech recognition,” ICASSP, 2021.
4. D. K. Hammond et al., “Wavelets on graphs via spectral graph theory,” Applied and Computational Harmonic Analysis, 2011.
5. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, 2011.
6. Add the full citation for the EEGIS dataset used by the project.

---

**Note:** This is a research prototype, not a clinically validated communication device. Predictions depend on the dataset, trained model, and evaluation setup.
