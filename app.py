
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import ui


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Envisioned Speech Decoder",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "eeg_app"

ui.apply_theme()
ui.show_header()

st.info(
    "Prototype mode: predictions use saved EEGIS trials, "
    "not live EEG recordings."
)


# ==================================================
# 2. LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(feature_path, label_path, vocab_path, raw_path):
    X = np.load(feature_path)
    y = np.load(label_path)
    vocab = joblib.load(vocab_path)
    raw_X = np.load(raw_path, mmap_mode="r")
    return X, y, vocab, raw_X


try:
    model = load_model(str(DATA_DIR / "model.pkl"))

    X, y, vocab, raw_X = load_data(
        str(DATA_DIR / "test_features.npy"),
        str(DATA_DIR / "test_labels.npy"),
        str(DATA_DIR / "vocab.pkl"),
        str(DATA_DIR / "raw_trials.npy")
    )

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.write("Expected data folder:", str(DATA_DIR))
    st.stop()

except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()


# ==================================================
# 3. VALIDATE DATA
# ==================================================

if X.ndim != 2 or X.shape[1] != 756:
    st.error(f"Expected 756 features per trial. Found {X.shape}.")
    st.stop()

if y.ndim != 1 or len(y) != len(X):
    st.error(
        "test_labels.npy must contain one label "
        "for every row in test_features.npy."
    )
    st.stop()

if raw_X.ndim != 4 or raw_X.shape[1:] != (128, 14, 6):
    st.error(
        f"Unexpected raw EEG shape: {raw_X.shape}. "
        "Expected (trials, 128, 14, 6)."
    )
    st.stop()

if len(X) == 0 or len(raw_X) == 0:
    st.error("No EEG trials were found.")
    st.stop()

expected_vocab = {
    0: "Rest",
    1: "Ayuda / Help",
    2: "Baño / Bath",
    3: "Dolor / Pain",
    4: "Gracias / Thank you",
    5: "Hambre / Hunger",
    6: "No",
    7: "Sed / Thirst",
    8: "Sí / Yes"
}

if not isinstance(vocab, dict):
    st.error("vocab.pkl must contain a class-ID dictionary.")
    st.stop()

if vocab != expected_vocab:
    st.warning(
        "The vocabulary differs from the expected EEGIS "
        "class mapping. Check vocab.pkl."
    )

max_trial = min(len(X), len(y), len(raw_X)) - 1


# ==================================================
# 4. SIDEBAR
# ==================================================

with st.sidebar:
    ui.show_sidebar_brand()

    st.subheader("Trial Settings")

    mode = st.selectbox(
        "Input type",
        ["Dataset EEG trial", "Noisy EEG trial"]
    )

    idx = st.slider(
        "Select trial",
        min_value=0,
        max_value=max_trial,
        value=0
    )

    noise = st.slider(
        "Feature noise level",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.05
    )

    channel_names = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ]

    channel = st.selectbox(
        "EEG channel",
        options=list(range(14)),
        format_func=lambda c: channel_names[c]
    )

    band_names = [
        "Raw", "Delta", "Theta", "Alpha", "Beta", "Gamma"
    ]

    band = st.selectbox(
        "Frequency representation",
        options=band_names
    )

    st.caption("EEGIS saved dataset · 128 Hz sampling")


# ==================================================
# 5. PREPARE SIMULATED INPUT
# ==================================================

signal = X[idx].astype(np.float64, copy=True)

if mode == "Noisy EEG trial" and noise > 0:
    # Reproducible noise for the selected trial and settings
    rng = np.random.default_rng(
        seed=idx + int(noise * 10000)
    )
    signal += rng.normal(0, noise, signal.shape)

band_index = band_names.index(band)

eeg_signal = np.asarray(
    raw_X[idx, :, channel, band_index],
    dtype=np.float64
)


# ==================================================
# 6. OVERVIEW CARDS
# ==================================================

ui.show_section(
    "Dataset Overview",
    "Summary of the loaded EEG data and current selection."
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    ui.show_card("EEG Trials", f"{len(X):,}")

with c2:
    ui.show_card("Features per Trial", X.shape[1])

with c3:
    ui.show_card("EEG Channels", "14")

with c4:
    ui.show_card("Classes", len(np.unique(y)))

st.write("")


# ==================================================
# 7. MAIN TABS
# ==================================================

overview_tab, signal_tab, prediction_tab, insights_tab = st.tabs([
    "Overview",
    "Signal Explorer",
    "Prediction",
    "Model Insights"
])


# ==================================================
# 8. OVERVIEW TAB
# ==================================================

with overview_tab:
    ui.show_section(
        "Project Overview",
        "Explore the EEG dataset and the simulated speech-classification workflow."
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Processing Pipeline")

        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #D8E5F4;
            border-radius: 14px;
            padding: 20px;
            line-height: 2.2;
        ">
            <b style="color:#1768C4;">01</b> &nbsp; Select saved EEG trial
            <br>
            <b style="color:#1768C4;">02</b> &nbsp; Explore signal channels
            <br>
            <b style="color:#1768C4;">03</b> &nbsp; Extracted features
            <br>
            <b style="color:#1768C4;">04</b> &nbsp; Model classification
            <br>
            <b style="color:#1768C4;">05</b> &nbsp; Predicted word candidate
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("### Dataset Details")

        ui.show_card(
            "Sampling Rate",
            "128 Hz",
            "Samples recorded per second"
        )

        st.write("")

        ui.show_card(
            "Trial Length",
            "128 samples",
            "Stored waveform length"
        )

    st.markdown("### Available Classes")

    class_columns = st.columns(3)

    for i, class_id in enumerate(sorted(expected_vocab)):
        with class_columns[i % 3]:
            ui.show_card(
                f"Class {class_id}",
                expected_vocab[class_id]
            )


# ==================================================
# 9. SIGNAL EXPLORER TAB
# ==================================================

with signal_tab:
    ui.show_section(
        "EEG Signal Explorer",
        "Inspect a stored EEG waveform by channel and frequency representation."
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        ui.show_card("Selected Trial", idx)

    with m2:
        ui.show_card("Selected Channel", channel_names[channel])

    with m3:
        ui.show_card("Representation", band)

    st.write("")

    if not np.all(np.isfinite(eeg_signal)):
        st.warning("The selected EEG waveform contains invalid values.")

    else:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            np.arange(len(eeg_signal)),
            eeg_signal,
            color="#1768C4",
            linewidth=1.7
        )

        ax.set_title(
            f"{band} Band — {channel_names[channel]}",
            color="#173E76",
            fontsize=14
        )
        ax.set_xlabel("Time Sample")
        ax.set_ylabel("Signal Amplitude")
        ax.set_xlim(0, len(eeg_signal) - 1)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        s1, s2, s3 = st.columns(3)

        with s1:
            ui.show_card("Samples", len(eeg_signal))

        with s2:
            ui.show_card("Minimum", f"{np.min(eeg_signal):.5g}")

        with s3:
            ui.show_card("Maximum", f"{np.max(eeg_signal):.5g}")

        with st.expander("View EEG sample values"):
            st.dataframe(
                pd.DataFrame({
                    "Time Sample": np.arange(len(eeg_signal)),
                    "Amplitude": eeg_signal
                }),
                use_container_width=True,
                hide_index=True
            )

    st.caption(
        "This is a stored EEGIS waveform, not a live EEG recording."
    )


# ==================================================
# 10. PREDICTION TAB
# ==================================================

with prediction_tab:
    ui.show_section(
        "Speech Prediction",
        "Generate a predicted class from the selected trial's feature vector."
    )

    st.caption(
        "Dataset Trial uses saved features. Noisy EEG trial adds "
        "simulated noise to the feature vector."
    )

    if st.button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    ):
        try:
            if not hasattr(model, "predict_proba"):
                st.error(
                    "The loaded model does not support predict_proba()."
                )
            elif not np.all(np.isfinite(signal)):
                st.error("Input features contain invalid values.")
            else:
                probs = model.predict_proba(
                    signal.reshape(1, -1)
                )[0]

                classes = model.classes_
                best = int(np.argmax(probs))
                predicted_class = int(classes[best])
                confidence = float(probs[best] * 100)
                predicted_word = vocab.get(
                    predicted_class,
                    f"Class {predicted_class}"
                )

                st.session_state["eeg_prediction"] = {
                    "trial": idx,
                    "mode": mode,
                    "noise": noise,
                    "word": predicted_word,
                    "class_id": predicted_class,
                    "confidence": confidence,
                    "classes": np.asarray(classes).tolist(),
                    "probabilities": probs.tolist()
                }

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    result = st.session_state.get("eeg_prediction")

    if result is not None:
        same_input = (
            result["trial"] == idx
            and result["mode"] == mode
            and result["noise"] == noise
        )

        if not same_input:
            st.info(
                "Settings changed. Generate a new prediction to update the result."
            )
        else:
            st.markdown("### Prediction Result")

            ui.show_card(
                "Predicted Word Candidate",
                result["word"],
                f"Class ID: {result['class_id']}"
            )

            st.write("")

            c1, c2 = st.columns(2)

            with c1:
                ui.show_card(
                    "Model Confidence",
                    f"{result['confidence']:.2f}%",
                    "Estimated probability for the selected class"
                )

            with c2:
                ui.show_card(
                    "Input Mode",
                    result["mode"]
                )

            st.write("")
            st.markdown("### Class Probabilities")

            chart_df = pd.DataFrame({
                "Class": [
                    vocab.get(int(class_id), f"Class {class_id}")
                    for class_id in result["classes"]
                ],
                "Probability (%)": (
                    np.asarray(result["probabilities"]) * 100
                )
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                chart_df.set_index("Class"),
                y="Probability (%)",
                y_label="Probability (%)",
                color="#2878D0"
            )

            with st.expander("View probability details"):
                st.dataframe(
                    chart_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Model confidence is not proof of conscious intention "
                "or a guarantee that the predicted word is correct."
            )

    else:
        st.info("Click Generate Prediction to view the predicted class.")


# ==================================================
# 11. MODEL INSIGHTS TAB
# ==================================================

with insights_tab:
    ui.show_section(
        "Model Insights",
        "Review the model and the arrays loaded by the application."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        ui.show_card("Model Type", type(model).__name__)

    with c2:
        ui.show_card("Feature Dimensions", f"{X.shape[1]}")

    with c3:
        ui.show_card(
            "Probability Support",
            "Available" if hasattr(model, "predict_proba") else "Unavailable"
        )

    st.markdown("### Data Shapes")

    st.dataframe(
        pd.DataFrame({
            "Data": [
                "Test features",
                "Test labels",
                "Raw EEG trials"
            ],
            "Shape": [
                str(X.shape),
                str(y.shape),
                str(raw_X.shape)
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    if hasattr(model, "classes_"):
        st.markdown("### Model Class IDs")
        st.write(list(model.classes_))

    st.info(
        "These saved dataset arrays do not establish live EEG decoding "
        "performance. Model evaluation requires a separate, correctly "
        "aligned test procedure."
    )


# ==================================================
# 12. FOOTER
# ==================================================

ui.show_footer()
