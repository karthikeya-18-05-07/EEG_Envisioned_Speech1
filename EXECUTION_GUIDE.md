# Execution Guide — Envisioned Speech Decoder

## 1. Add the actual data/model files
Copy these files from your working project into `eeg_app/`:
- `model.pkl`
- `test_features.npy`
- `test_labels.npy`
- `vocab.pkl`
- `raw_trials.npy`

Expected dimensions:
- `test_features.npy`: `(number_of_trials, 756)`
- `test_labels.npy`: `(number_of_trials,)`
- `raw_trials.npy`: `(number_of_trials, 128, 14, 6)`

The binaries were not supplied with the PowerPoint, so this ZIP contains source and documentation only.

## 2. Create the environment (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.12 is recommended because the saved model was reported as created with scikit-learn 1.6.1. A different scikit-learn version can cause pickle compatibility warnings or failures.

## 3. Start the app
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 4. Demonstration
1. Select a trial and EEG channel in the sidebar.
2. Select Raw, Delta, Theta, Alpha, Beta, or Gamma for waveform inspection.
3. Open **Signal Explorer** to view the selected stored waveform.
4. Open **Prediction** and click **Generate Prediction**.
5. The gateway displays a decoded word only if predicted class is not Rest and confidence is at least 40%; otherwise output is suppressed.
6. **Model Insights** shows model type and array shapes.

## 5. Scope
The Review 2 presentation reports the research pipeline (EA, 756 features, first-order graph smoothing, feature selection, SVM + Extra Trees soft voting, and preliminary evaluation). This app is the saved-artifact demonstration interface; it does not retrain that full pipeline at startup.

## 6. Reported results and status
Preliminary results in the Review 2 presentation: 78.66% mean accuracy and 78.70% mean F1 on stratified five-fold CV; mean subject-dependent accuracy 81.36%. LOSO, ablations, Chebyshev SGWT (K=50), runtime measurement, and several other experiments remain pending/proposed. Do not present pending experiments as completed results.

## 7. Troubleshooting
- **Required file not found:** check filenames and ensure all five files are inside `eeg_app/`.
- **Expected 756 features:** confirm the model feature matrix has 756 columns.
- **Unexpected raw EEG shape:** confirm the raw array shape is `(trials, 128, 14, 6)`.
- **Model loading error/version warning:** use the compatible scikit-learn version and Python environment used for training if available.
- **Prediction fails:** verify model and feature files came from the same trained pipeline and class mapping.

## 8. Before LMS submission
- Include the real model/data binaries if permitted by the dataset license.
- Add the official EEGIS dataset citation.
- Run the app using the actual files and verify all tabs and prediction.
- Do not claim live EEG input, LOSO results, Chebyshev SGWT, or runtime under 25 seconds unless those have been implemented and measured.
