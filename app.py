import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
from tensorflow.keras.models import load_model
from supabase import create_client
import uuid

# -----------------------
# CONFIG
# -----------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_tf_model():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tf_model()

# -----------------------
# PREPROCESS
# -----------------------
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    return data

def predict_image(image):
    data = preprocess_image(image)
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence = float(prediction[0][index])
    return class_name, confidence

# -----------------------
# UPLOAD SECTION
# -----------------------
st.title("AI Image Classifier Dashboard")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    with st.spinner("Processing..."):
        class_name, confidence = predict_image(image)

        # Upload to Supabase Storage
        file_id = str(uuid.uuid4()) + ".jpg"
        file_bytes = uploaded_file.getvalue()

        supabase.storage.from_("images").upload(file_id, file_bytes)

        public_url = supabase.storage.from_("images").get_public_url(file_id)

        # Insert into DB
        supabase.table("images").insert({
            "image_path": public_url,
            "predicted_class": class_name,
            "confidence_score": confidence
        }).execute()

    st.success(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence:.2f}")

# -----------------------
# FILTER + DISPLAY
# -----------------------
st.divider()
st.header("Browse Images")

response = supabase.table("images").select("*").execute()
rows = response.data

if rows:
    classes = sorted(list(set(r["predicted_class"] for r in rows)))
    selected_class = st.selectbox("Filter by class", ["All"] + classes)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5)

    filtered = [
        r for r in rows
        if (selected_class == "All" or r["predicted_class"] == selected_class)
        and r["confidence_score"] >= min_conf
    ]

    cols = st.columns(3)

    for i, item in enumerate(filtered):
        with cols[i % 3]:
            st.image(item["image_path"])
            st.write(item["predicted_class"])
            st.write(f"{item['confidence_score']:.2f}")
else:
    st.info("No images yet.")
