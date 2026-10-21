from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(
    page_title="YOLO26n 物件偵測",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 YOLO26n 物件偵測系統")
st.write("上傳圖片或使用相機拍照，辨識圖片中的物件。")

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "找不到 best.pt。請確認已將訓練好的模型檔案放在 app.py 同一個資料夾。"
        )
    return YOLO(str(MODEL_PATH))

confidence = st.slider(
    "偵測信心分數門檻",
    min_value=0.10,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

source = st.radio(
    "選擇影像來源",
    ["上傳圖片", "相機拍照"],
    horizontal=True,
)

if source == "上傳圖片":
    image_file = st.file_uploader(
        "請選擇圖片",
        type=["jpg", "jpeg", "png"],
    )
else:
    image_file = st.camera_input("請拍攝要辨識的物件")

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="原始圖片", use_container_width=True)

        if st.button("開始辨識", type="primary"):
            with st.spinner("模型辨識中，請稍候..."):
                model = load_model()
                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("辨識結果")
                st.image(
                    result.plot()[:, :, ::-1],
                    caption="物件偵測框",
                    use_container_width=True,
                )

            with col2:
                st.subheader("偵測清單")
                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())
                        detections.append({
                            "物件": result.names[class_id],
                            "信心分數": f"{score:.1%}",
                        })

                if detections:
                    st.dataframe(
                        detections,
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.metric("偵測物件總數", len(detections))
                else:
                    st.info("沒有偵測到符合門檻的物件。")

    except Exception as error:
        st.error(f"辨識時發生錯誤：{error}")

st.caption(
    "本系統使用既有 YOLO26n 模型。辨識結果僅供參考；"
    "請勿單憑模型結果判斷物品是否具有危險性。"
)
