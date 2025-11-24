import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = r"C:\Users\HP\Desktop\job"
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
MASKS_DIR = os.path.join(BASE_DIR, "dataset", "masks")  # where your mask images are

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Function to compute line lengths from mask
# -----------------------------
def line_lengths(mask):
    # Count pixels in color ranges (BGR)
    life = np.sum(cv2.inRange(mask, (0, 0, 150), (50, 50, 255)))   # red
    head = np.sum(cv2.inRange(mask, (0, 150, 150), (50, 255, 255))) # yellow
    heart = np.sum(cv2.inRange(mask, (150, 0, 0), (255, 50, 50)))   # blue
    return life, head, heart

# -----------------------------
# Compute features for all dataset masks
# -----------------------------
life_list, head_list, heart_list = [], [], []
for img_id in df['image_id']:
    mask_path = os.path.join(MASKS_DIR, f"{img_id}_mask.png")
    if not os.path.exists(mask_path):
        life_list.append(0)
        head_list.append(0)
        heart_list.append(0)
    else:
        mask = cv2.imread(mask_path)
        l, h, he = line_lengths(mask)
        life_list.append(l)
        head_list.append(h)
        heart_list.append(he)

df['life_len'] = life_list
df['head_len'] = head_list
df['heart_len'] = heart_list

# -----------------------------
# Train RandomForest Classifier
# -----------------------------
X = df[['life_len', 'head_len', 'heart_len']].values
y = df['dominant_line']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Palm-Astro Line Predictor")
st.write(f"Model Accuracy on Test Data: **{acc*100:.2f}%**")
st.write("Upload a **mask image** of the palm (red=Life, yellow=Head, blue=Heart)")

uploaded_file = st.file_uploader("Choose a mask image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    mask = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Compute features
    life_len, head_len, heart_len = line_lengths(mask)
    st.write(f"Line Lengths: Life={life_len}, Head={head_len}, Heart={heart_len}")

    # Predict dominant line
    pred = clf.predict([[life_len, head_len, heart_len]])
    st.success(f"Predicted Dominant Line: {pred[0]}")

    # Show the uploaded mask
    st.image(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), caption="Uploaded Mask", use_column_width=True)
