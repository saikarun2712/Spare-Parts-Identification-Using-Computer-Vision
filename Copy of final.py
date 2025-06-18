import os
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from google.cloud import vision
from fuzzywuzzy import fuzz
import io

# Paths to resources
RAM_FILE_PATH = "/Users/sravya/Desktop/Part Master File/RAMs.csv"
MOTHERBOARD_FILE_PATH = "/Users/sravya/Desktop/Part Master File/MBs.csv"
MODEL_PATH = "/Users/sravya/Desktop/runs/detect/train/weights/best.pt"
GCLOUD_CREDENTIALS_PATH = "/Users/sravya/Desktop/cred.json"

# Set Google Cloud credentials
def set_gcloud_credentials():
    if not os.path.exists(GCLOUD_CREDENTIALS_PATH):
        st.error("Error: Google Cloud credentials file not found.")
        return False
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCLOUD_CREDENTIALS_PATH
    return True

# Google Cloud Vision API integration
def detect_text_with_gcloud(image):
    """Use Google Cloud Vision API to detect text in an image."""
    client = vision.ImageAnnotatorClient()
    _, image_bytes = cv2.imencode('.jpg', image)
    image_content = image_bytes.tobytes()
    vision_image = vision.Image(content=image_content)

    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    if response.error.message:
        st.error(f"Error with Google Cloud Vision API: {response.error.message}")
        return "Error"

    return texts[0].description if texts else "No text found"

# Check detected text against reference CSV
def check_reference(detected_text, csv_path, threshold=80):
    """Match detected text against the 'White Label' column in the reference CSV."""
    df = pd.read_csv(csv_path)

    best_match = None
    highest_score = 0

    # Split the detected text into words or small phrases
    detected_segments = detected_text.split()

    for segment in detected_segments:
        # Compare each detected segment with the 'White Label' column
        for _, row in df.iterrows():
            reference_value = str(row['White Label']).lower().strip()
            score = fuzz.partial_ratio(segment.lower(), reference_value)

            # If the score is higher than the previous best match, update the best match
            if score > highest_score:
                highest_score = score
                best_match = row

    # Return the best match as a table or error message
    if highest_score >= threshold and best_match is not None:
        matched_table = pd.DataFrame([best_match])
        st.table(matched_table[['DHL Given Name','White Label', 'Product Number', 'Company', 'Customer Product Name']])
        st.success(f"Matching part found with confidence: {highest_score}%")
    else:
        st.warning("No matching part found in the Part Master File. Please check the image or try again.")

# Process image using Google Cloud Vision API
def process_google_vision(image):
    """Process an image with Google Cloud Vision and match against the RAM CSV."""
    st.info("Processing the image with Google Cloud Vision API.")
    detected_text = detect_text_with_gcloud(image)
    st.info("The text from the label is extracted.")
    if detected_text != "Error":
        check_reference(detected_text, RAM_FILE_PATH)

# Object detection with YOLOv8
def detect_obj(image):
    """Detect objects using YOLOv8 and decide between CSV matching and Google Vision."""
    model = YOLO(MODEL_PATH)
    st.info("YOLOv8 model deployed.")
    results = model.predict(image)

    if not results or len(results[0].boxes) == 0:
        st.warning("No object was detected. Please try uploading a clearer image.")
        return False

    ram_data, motherboard_data = load_data(for_google_api=False)
    if ram_data is None or motherboard_data is None:
        return False

    for box in results[0].boxes:
        class_id = int(box.cls)
        name = model.names[class_id]
        confidence = box.conf.item() 
        bbox = box.xyxy[0].numpy()

       # Confidence handling
        if "memory" in name.lower():  # Adjusted to ensure case-insensitivity
            if "unlabelled" in name.lower():  
                if 0.69 < confidence < 0.85:
                    st.error(f"LOW CONFIDENCE: {confidence:.2f}")
                    st.error(" THIS PART IS UNLABELED.")
                    st.error("Please upload the part with a label or a clearer image of the part.")
                elif confidence >= 0.85:
                    st.warning(f"HIGH CONFIDENCE: {confidence:.2f}")
                else:
                    st.error(f"LOW CONFIDENCE: {confidence:.2f}")
                    st.error("SPARE PART UNKNOWN")
            else:  # For labelled memory parts
                if 0.69 < confidence < 0.85:
                    st.error(f"LOW CONFIDENCE: {confidence:.2f} ")
                    st.warning("Deploying Google Cloud Vision API for further analysis.")
                    process_google_vision(image)
                elif confidence >= 0.85:
                    st.success(f"HIGH CONFIDENCE: {confidence:.2f}.")
                    st.success(f"Object recognized as {name}.")
                    labeled_image = draw_bounding_box(image, bbox, name, confidence)
                    st.image(labeled_image, caption=f"Detected: {name} with confidence {confidence:.2f}", use_column_width=True)
                    check_reference(name, RAM_FILE_PATH)
                else:
                    st.error(f"LOW CONFIDENCE: ({confidence:.2f})")
                    st.error("SPARE PART UNKNOWN")

        else:  # Non-RAM case (e.g., motherboard)
            if 0.69 < confidence < 0.85:
                st.error(f"LOW CONFIDENCE: ({confidence:.2f})")
                st.error("NO LABEL FOUND.")
                st.error("Please upload a clearer image of the part.")
            elif confidence >= 0.85:
                st.success(f"HIGH CONFIDENCE: {confidence:.2f}.")
                st.sucess(f"Object recognized as {name}.")
                labeled_image = draw_bounding_box(image, bbox, name, confidence)
                st.image(labeled_image, caption=f"Detected: {name} with confidence {confidence:.2f}", use_column_width=True)
                check_reference(name, MOTHERBOARD_FILE_PATH)
            else:
                st.error(f"LOW CONFIDENCE: ({confidence:.2f})")
                st.error("SPARE PART UNKNOWN")


    return True

# Load CSV data dynamically
def load_data(for_google_api=False):
    """Load RAM and/or Motherboard data based on the detection method."""
    if for_google_api:
        if not os.path.exists(RAM_FILE_PATH):
            st.error("Error: RAM Part Master File could not be found.")
            return None
        return pd.read_csv(RAM_FILE_PATH), None
    else:
        if not (os.path.exists(RAM_FILE_PATH) and os.path.exists(MOTHERBOARD_FILE_PATH)):
            st.error("Error: Part Master File(s) could not be found.")
            return None, None
        ram_data = pd.read_csv(RAM_FILE_PATH)
        motherboard_data = pd.read_csv(MOTHERBOARD_FILE_PATH)
        return ram_data, motherboard_data

# Draw bounding box
def draw_bounding_box(image, bbox, label, confidence):
    """Draw bounding box with label and confidence score."""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = get_font(18)
    x1, y1, x2, y2 = bbox
    box_color = "red"
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
    text = f"{label} {confidence:.2f}"
    draw.text((x1, y1 - 10), text, fill="white", font=font)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

# Streamlit App UI
def main():
    st.title("Spare Part Detection with YOLOv8 and Google Cloud Vision")
    st.write("Upload an image, and click 'IDENTIFY' to detect the spare part.")

    if not set_gcloud_credentials():
        st.error("Unable to set Google Cloud credentials.")
        return

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if st.button("IDENTIFY"):
            detect_obj(image_cv)
    else:
        st.write("Please upload an image file.")

if __name__ == "__main__":
    main()
