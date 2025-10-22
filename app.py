import streamlit as st
from PIL import Image
import numpy as np
import os
from keras.applications.xception import Xception
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from pickle import load
import gdown

# --------------------------
# Paths & Models
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATASET_DIR = os.path.join(BASE_DIR, "Flicker8k_Dataset")
DATASET_DIR = "data/Flicker8k_Dataset"
FEATURE_MODEL = Xception(include_top=False, pooling="avg")

# Load tokenizer
tokenizer = load(open(os.path.join(MODEL_DIR, "tokenizer.p"), "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = 32  # same as training

# --------------------------
# Define Captioning Model
# --------------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

caption_model = define_model(vocab_size, max_length)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file ID
file_id = "1ulCGMmQ9CVRpuVqA-XMXi-6Byzp1CyiY"
model_path = os.path.join(MODEL_DIR, "model_9.h5")

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load weights
caption_model.load_weights(model_path)

# --------------------------
# Utilities
# --------------------------
def extract_features(image, model):
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# --------------------------
# Beam Search Caption Generation
# --------------------------
def generate_desc_beam_search(model, tokenizer, photo, max_length, beam_index=3):
    start = [([tokenizer.word_index['start']], 0.0)]
    
    while len(start[0][0]) < max_length:
        temp = []
        for s in start:
            sequence, score = s
            sequence_padded = pad_sequences([sequence], maxlen=max_length)
            preds = model.predict([photo, sequence_padded], verbose=0)[0]
            top_indices = preds.argsort()[-beam_index:][::-1]
            for idx in top_indices:
                word = word_for_id(idx, tokenizer)
                if word is None:
                    continue
                new_seq = sequence + [idx]
                new_score = score - np.log(preds[idx] + 1e-10)
                temp.append((new_seq, new_score))
        start = sorted(temp, key=lambda t: t[1])[:beam_index]
    
    best_sequence = start[0][0]
    caption_words = [word_for_id(idx, tokenizer) for idx in best_sequence]
    caption = ' '.join([w for w in caption_words if w not in ['start', 'end']])
    return caption

# --------------------------
# Streamlit App
# --------------------------
st.title("🖼️ Image Caption Generator")

st.info(
    "⚠️ Note: This model was trained on the **Flickr8k dataset**, which mostly contains everyday scenes, people, and crowds. "
    "For best results, please upload images **similar to Flickr8k** (e.g., people, animals, outdoor activities). "
    "Images outside this domain may produce generic captions."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Generating caption...")

    photo = extract_features(image, FEATURE_MODEL)
    caption = generate_desc_beam_search(caption_model, tokenizer, photo, max_length, beam_index=3)
    
    st.success("Caption generated!")
    st.write(f"**Caption:** {caption}")

# --------------------------
# Show Example Flickr8k Images
# --------------------------
st.subheader("📸 Example images from Flickr8k")
st.write(
    "Try to upload images similar to these for the best caption results:"
)

example_images = [
    "1859941832_7faf6e5fa9.jpg",
    "10815824_2997e03d76.jpg",
    "3759492488_592cd78ed1.jpg",
    "3724581378_41049da264.jpg"
]

cols = st.columns(len(example_images))
for col, img_name in zip(cols, example_images):
    img_path = os.path.join(DATASET_DIR, img_name)
    if os.path.exists(img_path):
        example_img = Image.open(img_path)
        example_img = example_img.resize((200, 200))
        col.image(example_img, use_container_width=True)
