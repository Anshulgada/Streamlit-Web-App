import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Custom CSS
st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

# Hide Streamlit's menu and footer
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div.block-container {padding-top: 1rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Pothole Detection")

# Path to your YOLOv10 model weights
model_url = 'https://github.com/Anshulgada/Streamlit-Web-App/raw/main/best.pt'

# Load YOLOv10 model
model = YOLO(model_url)

# Function to perform inference on uploaded image
def run_inference():
    # Run inference on an image
    results = model([uploaded_file])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Key points object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen

    # return results

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True, width=400)
        if st.button('Run Inference'):
            st.write("Classifying...")
            # Perform inference on the uploaded image
            results = run_inference()

    with col2:
        if 'results' in locals():
            # Display processed image with bounding boxes
            for result in results:
                img_with_boxes = Image.fromarray(result)
                st.image(img_with_boxes, caption='Processed Image.', use_column_width=True, width=400)
