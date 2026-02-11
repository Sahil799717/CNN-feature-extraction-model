import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from PIL import Image

st.set_page_config(page_title="CNN Feature Visualizer", layout="wide")

st.title("🧠 VGG16 CNN Feature Visualizer")
st.markdown("Upload an image to visualize intermediate CNN feature maps.")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(inputs=base_model.input, outputs=outputs)

    return model, layer_names

model, layer_names = load_model()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✅ Uploaded Image", width=300)

    if st.button("🔍 Show Feature Maps"):

        with st.spinner("🚀 Processing..."):

            # Resize image
            image_resized = image.resize((224, 224))

            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            feature_maps = model.predict(img_array)

            st.success("🎯 Feature maps generated successfully!")

            # Display feature maps
            for layer_name, feature_map in zip(layer_names, feature_maps):

                st.markdown(f"### 🔬 Layer: `{layer_name}`")

                cols = st.columns(6)

                for i in range(min(feature_map.shape[-1], 6)):
                    fmap = feature_map[0, :, :, i]

                    # Normalize
                    fmap -= fmap.min()
                    fmap /= (fmap.max() + 1e-8)
                    fmap *= 255.0

                    cols[i].image(
                        fmap.astype(np.uint8),
                        use_container_width=True,
                        caption=f"Filter {i+1}"
                    )

                st.markdown("---")
