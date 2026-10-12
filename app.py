import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
APP_TITLE = "WildLens AI"
MODEL_PATH = Path(__file__).parent / "animal_classification_model.keras"
IMG_SIZE = (160, 160)

# IMPORTANT: Keep this order identical to class_names used during training.
# The order below matches sorted(df["class"].unique()) for the dataset described.
CLASS_NAMES = ["Bear", "Elephant", "Horse", "Lion", "Panda", "Tiger"]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#eaf0ff; --muted:#aab6d3; --panel:#111b31; --line:rgba(172,190,231,.16); --accent:#8b9cff; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, #1b3154 0%, #0a1020 42%, #080d18 100%); color:var(--ink); }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#101b31,#0a1120); border-right:1px solid var(--line); }
      .block-container { padding-top:2rem; padding-bottom:3rem; max-width:1400px; }
      .hero { padding:1.7rem 1.8rem; border:1px solid var(--line); border-radius:24px; background:linear-gradient(120deg,rgba(34,55,91,.92),rgba(17,27,49,.78)); margin-bottom:1.4rem; }
      .eyebrow { color:#a9b7ff; text-transform:uppercase; letter-spacing:.16em; font-size:.74rem; font-weight:800; }
      .hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(2.1rem,4vw,3.6rem); letter-spacing:-.045em; line-height:1.08; margin:.45rem 0 .8rem; color:#f5f7ff; }
      .hero p { color:#c1cce4; font-size:1.03rem; max-width:780px; margin-bottom:0; }
      .pill { display:inline-block; border:1px solid rgba(169,183,255,.28); border-radius:999px; padding:.32rem .7rem; color:#dce2ff; background:rgba(139,156,255,.1); font-size:.78rem; margin:0 .3rem .3rem 0; }
      .panel { background:rgba(17,27,49,.84); border:1px solid var(--line); border-radius:20px; padding:1.15rem 1.2rem; }
      .metric { background:rgba(27,42,71,.78); border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; min-height:104px; }
      .metric-label { color:var(--muted); font-size:.8rem; margin-bottom:.35rem; }
      .metric-value { color:#f6f8ff; font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; }
      .muted { color:var(--muted); }
      .small-note { color:#aab6d3; font-size:.82rem; }
      div[data-testid="stFileUploader"] { border:1px dashed rgba(169,183,255,.45); border-radius:16px; padding:.5rem; background:rgba(14,23,41,.6); }
      div.stButton > button, div.stDownloadButton > button { border-radius:12px; border:1px solid rgba(169,183,255,.35); min-height:2.8rem; font-weight:700; }
      div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#7b8cff,#a18bff); color:#0b1020; border:none; }
      div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#7b8cff,#69d9c8); }
      .result-title { font-size:1.85rem; font-weight:800; color:#f6f8ff; font-family:'Manrope',sans-serif; }
      .stTabs [data-baseweb="tab-list"] { gap:.5rem; }
      .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.6rem 1rem; }
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Model loading - cached to avoid reloading on every interaction
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained CNN model…")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def prepare_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Apply the same aspect-ratio-preserving resize as notebook inference."""
    original = ImageOps.exif_transpose(image).convert("RGB")
    # resize_with_pad equivalent: preserve aspect ratio and pad to 160x160
    resized = ImageOps.pad(
        original,
        (IMG_SIZE[1], IMG_SIZE[0]),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    array = np.asarray(resized, dtype=np.float32)
    # The saved model includes Rescaling(1./255), so DO NOT divide by 255 here.
    return original, np.expand_dims(array, axis=0)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐾 WildLens AI")
    st.caption("CNN-powered animal image classification")
    st.divider()
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a clear animal image.\n"
        "2. Select **Classify image**.\n"
        "3. Review the predicted class and confidence scores."
    )
    st.divider()
    st.markdown("### Model information")
    st.markdown("**Architecture:** Custom CNN")
    st.markdown("**Input size:** 160 × 160 × 3")
    st.markdown("**Classes:** 6 animals")
    st.markdown("**Preprocessing:** RGB + aspect-preserving resize")
    st.caption("Predictions are estimates and can be wrong, especially for blurry, cropped, or out-of-scope images.")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Computer vision · Image classification</div>
      <h1>Meet your image.<br>Discover its animal.</h1>
      <p>Upload an image and let your trained convolutional neural network estimate which of six animal classes it belongs to.</p>
      <div style="margin-top:1rem">
        <span class="pill">🐻 Bear</span><span class="pill">🐘 Elephant</span>
        <span class="pill">🐎 Horse</span><span class="pill">🦁 Lion</span>
        <span class="pill">🐼 Panda</span><span class="pill">🐅 Tiger</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Guard against missing model / class mismatch
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Put `animal_classification_model.keras` "
        "in the same folder as `app.py`, or update `MODEL_PATH` near the top of the file."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Check that the file is a valid Keras model and TensorFlow is installed.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

if model.input_shape[-3:] != (160, 160, 3):
    st.warning(
        f"This app expects a model input of (160, 160, 3), but the loaded model reports {model.input_shape}. "
        "Update IMG_SIZE and preprocessing to match the model you trained."
    )

# ------------------------------------------------------------
# Main upload / prediction flow
# ------------------------------------------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("### 01 · Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, PNG, or WEBP image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="For best results, use one clearly visible animal with good lighting.",
        accept_multiple_files=False,
    )

    if uploaded_file:
        try:
            uploaded_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            original_image, image_batch = prepare_image(uploaded_image)
            st.image(original_image, caption=f"Preview · {original_image.width} × {original_image.height}", use_container_width=True)
            st.caption("The preview is the original image. The model receives a padded 160 × 160 RGB version.")
        except Exception as exc:
            st.error("This file could not be opened as an image. Try a different JPG or PNG.")
            with st.expander("Technical details"):
                st.code(str(exc))
            uploaded_file = None

    classify = st.button(
        "✨ Classify image",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

with right:
    st.markdown("### 02 · Prediction")
    if classify and uploaded_file is not None:
        try:
            with st.spinner("Analyzing image…"):
                # Re-prepare from upload bytes at prediction time to avoid stale state.
                img = Image.open(io.BytesIO(uploaded_file.getvalue()))
                original_image, image_batch = prepare_image(img)
                probabilities = np.asarray(model.predict(image_batch, verbose=0)[0]).reshape(-1)

            if probabilities.size != len(CLASS_NAMES):
                st.error(
                    f"The model returned {probabilities.size} outputs, but CLASS_NAMES contains "
                    f"{len(CLASS_NAMES)} labels. Update CLASS_NAMES to match the exact training label order."
                )
                st.stop()

            best_index = int(np.argmax(probabilities))
            best_class = CLASS_NAMES[best_index]
            best_probability = float(probabilities[best_index])

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Top prediction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">{best_class}</div>', unsafe_allow_html=True)
            st.progress(min(max(best_probability, 0.0), 1.0), text=f"Model score · {format_percent(best_probability)}")
            if best_probability < 0.50:
                st.warning("The model is not very confident. Try a clearer image with the animal centered.")
            elif best_probability < 0.75:
                st.info("Moderate confidence. Check the other class scores below.")
            else:
                st.success("High model score. This is still an estimate, not a guarantee.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Class score breakdown")
            order = np.argsort(probabilities)[::-1]
            for idx in order:
                st.write(f"**{CLASS_NAMES[int(idx)]}** · {format_percent(float(probabilities[idx]))}")
                st.progress(min(max(float(probabilities[idx]), 0.0), 1.0))

            st.download_button(
                "⬇️ Download prediction summary",
                data=(
                    "WildLens AI — Prediction Summary\n"
                    f"Image: {uploaded_file.name}\n"
                    f"Predicted class: {best_class}\n"
                    f"Model score: {format_percent(best_probability)}\n\n"
                    "Class scores:\n"
                    + "\n".join(
                        f"{CLASS_NAMES[int(idx)]}: {format_percent(float(probabilities[idx]))}"
                        for idx in order
                    )
                    + "\n\nNote: Model scores are estimates and may not equal real-world correctness."
                ),
                file_name="wildlens_prediction.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as exc:
            st.error("Prediction failed. Verify that the model's input size, preprocessing, and class order match the notebook.")
            with st.expander("Technical details"):
                st.code(str(exc))
    else:
        st.markdown(
            """
            <div class="panel">
              <div class="eyebrow">Ready when you are</div>
              <div class="result-title" style="font-size:1.35rem">Your result will appear here</div>
              <p class="muted">Upload an image on the left, then select <b>Classify image</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("**Tips for a better prediction**")
        st.markdown(
            "- Show the animal clearly and avoid heavy blur.\n"
            "- Use a well-lit image where the animal is not too small.\n"
            "- The model only knows the six classes listed above."
        )

st.divider()
st.markdown(
    '<p class="small-note">WildLens AI · Educational CNN demo · Your uploaded image is processed by this app for prediction; '
    'results depend on the training data and model quality.</p>',
    unsafe_allow_html=True,
)
