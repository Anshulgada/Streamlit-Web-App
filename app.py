import streamlit as st
import requests
from PIL import Image, ImageDraw
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Pothole Detection System", page_icon=":camera:", layout="wide")

# Custom CSS for dark mode, styling, and hiding specific elements
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .stTextInput input, .stNumberInput input, .stFileUploader div, .stSelectbox div {
            background-color: #333;
            color: #ffffff;
        }
        .stButton button {
            background-color: #6200ea;
            color: #ffffff;
        }
        .stRadio div {
            color: #ffffff;
        }
        .stRadio>div>label {
            color: #ffffff;
        }
        .viewerBadge_container__1QSob { 
            display: none;
        }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        div.block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("RoboFlow Inference")

# Input form for model settings, upload method, and inference options
with st.form(key='inputForm'):
    st.header("Model Settings")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        model = st.text_input('Model', 'pothole-detection-system-cvgcc')

    with col2:
        version = st.number_input('Version', value=6, step=1)

    with col3:
        api_key = st.text_input('API Key', 'lzUEJC1DdDXWvK4nsl6K')

    st.header("Upload Method")
    method = st.radio('', ('Upload', 'URL'))

    uploaded_file = None
    url = None

    if method == 'Upload':
        uploaded_file = st.file_uploader('Select File', type=['jpg', 'jpeg', 'png', 'mp4'])
        if uploaded_file is not None:
            st.text('Selected file:')
            st.write(uploaded_file)
    else:
        url = st.text_input('Enter Image URL: ')

    st.header("Filter Settings")
    filter_classes = st.text_input('Filter Classes', help='Separate names with commas')

    min_confidence = st.slider('Min Confidence', 0, 100, 60)

    max_overlap = st.slider('Max Overlap', 0.0, 1.0, 0.5, step=0.05)

    st.subheader("Inference Result")
    inference_result = st.radio("Select Inference Result Format", ["Image", "JSON"])

    col6, col7 = st.columns(2)
    with col6:
        labels = st.radio("Labels", ["On", "Off"])
    with col7:
        stroke_width = st.radio("Stroke Width", ["1px", "2px", "5px", "10px"])

    submitted = st.form_submit_button('Run Inference')

if submitted:
    try:
        if uploaded_file is not None:
            files = {"file": uploaded_file.getvalue()}
            request_url = f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}"
        else:
            files = None
            request_url = f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}&image={url}"

        params = {
            "confidence": min_confidence / 100,
            "overlap": max_overlap / 100
        }

        if filter_classes:
            params["classes"] = filter_classes.split(',')

        response = requests.post(request_url, files=files, params=params)
        response.raise_for_status()
        results = response.json()

        if inference_result == "JSON":
            st.json(results)

        elif inference_result == "Image":
            image_url = results.get("image")
            if image_url:
                # Fetch image width and height from the response
                image_width = results['image']['width']
                image_height = results['image']['height']

                # Load image from URL
                response_image = requests.get(image_url)
                image = Image.open(BytesIO(response_image.content))

                # Create PIL ImageDraw object
                draw = ImageDraw.Draw(image)

                # Iterate over predictions and draw bounding boxes
                for prediction in results['predictions']:
                    x, y = prediction['x'], prediction['y']
                    width, height = prediction['width'], prediction['height']

                    x0, y0 = x - width / 2, y - height / 2
                    x1, y1 = x + width / 2, y + height / 2

                    draw.rectangle([x0, y0, x1, y1], outline="red", width=int(stroke_width[:-2]))

                    if labels == "On":
                        draw.text((x0, y0 - 10), prediction['class'], fill="red")

                # Display image with bounding boxes
                st.image(image, caption=f"Detected objects (width: {image_width}, height: {image_height})",
                         use_column_width=True)
            else:
                st.error("Image URL not found in the response.")

    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Placeholder for showing inference results
st.markdown("### Inference Results")

# Display the uploaded file or the URL if provided
if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)
elif url:
    st.image(url, use_column_width=True)