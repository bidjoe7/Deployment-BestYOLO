import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import numpy as np
from datetime import datetime
import base64
from moviepy.editor import VideoFileClip, ImageSequenceClip
import subprocess
import wget
from collections import Counter

st.set_page_config(layout="wide")

model = None
confidence = .25

def image_input(data_src):
    img_file = None
    
    if data_src == 'Sample Data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            ts = datetime.timestamp(datetime.now())
            img_file = os.path.join('data/uploaded_data', str(ts)+ img_bytes.name)
            Image.open(img_bytes).save(img_file)
    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img, counts = infer_image(img_file)
            st.image(img, caption="Model prediction")
            st.markdown('### Detection Counts')
            for label, count in counts.items():
                st.markdown(f'* **{label}**: {count}')
            #save prediction file in data/prediction_data path (Butuh penambahan folder)
            output_file = os.path.join('data/prediction_data', os.path.basename(img_file))
            img.save(output_file)
            #download prediction file
            with open(output_file, 'rb') as f:
                data = f.read()
                b64 = base64.b64encode(data).decode('UTF-8')
                href = f'<a href="data:image/jpeg;base64,{b64}" download="output.jpg"> 📥 Download Model Prediction</a>'
                st.markdown(href, unsafe_allow_html=True)

def video_input(data_src):
    vid_file = None
    if data_src == 'Sample Data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            ts = datetime.timestamp(datetime.now())
            vid_file = os.path.join('data/uploaded_data', str(ts)+ vid_bytes.name)
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
      col1, col2 = st.columns(2)
      with col1:
        #display uploaded video
        st_video = open(vid_file, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
      
        cap = cv2.VideoCapture(vid_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
        output = st.empty()
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Display Prediction Video...")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, labels = infer_image1(frame)  # Call the function with the added functionality
            result_array = np.array(output_img)
            results.append(result_array)

        # Count the number of occurrences of each label
        counts = Counter(labels)
        st.write(counts)  # Display the counts


        cap.release()

        prediction_clip = ImageSequenceClip(results, fps=fps)
        prediction_path = os.path.join('data/prediction_data', os.path.basename(vid_file))
        prediction_clip.write_videofile(prediction_path)
      with col2:  
        # Display prediction video in Streamlit
        st.video(prediction_path)
        st.write("Prediction Video")


#Webcam

#draw bounding box
def predict(model, frame):
    """Generate predictions and annotate the predicted frame."""
    model.conf = confidence
    resx = model(frame, size=416).crop(save=False) 
    for d in resx:
        box = list(map(int, list(map(np.round, d['box']))))
        label = d['label']
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
        px, py = box[0], box[1] - 10
        cv2.putText(frame, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class VideoProcessor:
    def recv(self, frame):
        # def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = predict(model, img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])

    # Get the labels of detected objects
    labels = [model.names[int(x[-1])] for x in result.xyxy[0]]
    counts = Counter(labels)
    return image, counts

def infer_image1(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])

    # Get the labels of detected objects
    labels = [model.names[int(x[-1])] for x in result.xyxy[0]]

    return image, labels


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('WangRongsheng/BestYOLO', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("👷 Safety Gear Detection Dashboard")

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("Select Model", ["👷🏻‍♂️ Safety Protocol", "🦺 Safety Vest", "⛑️ Helmet","👷🏾‍♂️ Safety Protocol [SCYRISLite]", "🦺 Safety Vest [SCYRISLite]", "⛑️ Helmet [SCYRISLite]"])
    # URL, upload file (max 200 mb)
    if model_src == "👷🏻‍♂️ Safety Protocol":
        url = "https://huggingface.co/BIDJOE/yolov5n-resnet50xSPPCSPCxGhostNet/resolve/main/Safety_protocol-best.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file
    if model_src == "🦺 Safety Vest":
        url = "https://huggingface.co/BIDJOE/yolov5n-resnet50xSPPCSPCxGhostNet/resolve/main/Safety_vest-best.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file
    if model_src == "⛑️ Helmet":
        url = "https://huggingface.co/BIDJOE/yolov5n-resnet50xSPPCSPCxGhostNet/resolve/main/Hard_hat-best.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file
    if model_src == "👷🏾‍♂️ Safety Protocol [SCYRISLite]":
        url = "https://huggingface.co/BIDJOE/SCYRISLite-SCYTRIS/resolve/main/BiFPN-SafetyProtocol.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file
    if model_src == "🦺 Safety Vest [SCYRISLite]":
        url = "https://huggingface.co/BIDJOE/yolov5n-resnet50xSPPCSPCxGhostNet/resolve/main/Safety_vest-best.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file
    if model_src == "⛑️ Helmet [SCYRISLite]":
        url = "https://huggingface.co/BIDJOE/SCYRISLite-SCYTRIS/resolve/main/BiFPN-HardHat.pt"
        model_file_ = download_model(url)
        if model_file_.split(".")[-1] == "pt":
            model_file = model_file_
            cfg_model_path = model_file

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.selectbox("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.selectbox("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # custom classes

        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0], model_names[1]])
        classes = [model_names.index(name) for name in assigned_class]
        model.classes = classes
        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select Input Type: ", ['🖼️ Image', '📽️ Video', '📷 Webcam'])

        # input src option
        data_src = st.sidebar.radio("Select Input Source: ", ['Sample Data', 'Upload Your Own Data'])

        if input_option == '🖼️ Image':
            image_input(data_src)
        elif input_option == '📽️ Video':
            video_input(data_src)
        else:
            webrtc_streamer(key="objectDetector", video_transformer_factory=VideoProcessor,
                        rtc_configuration=RTCConfiguration({
                            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                        }))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

