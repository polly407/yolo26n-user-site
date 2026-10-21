# YOLO26n 物件偵測網站

這個 Streamlit 網站使用既有的 YOLO 模型進行物件偵測，不會重新訓練模型。

## 功能
- 上傳 JPG、JPEG、PNG 圖片
- 使用裝置相機拍照
- 調整偵測信心分數門檻
- 顯示偵測框、類別與信心分數

## 模型檔案
請將訓練好的 `best.pt` 放在本 Repository 根目錄，與 `app.py` 同一層。

## 部署到 Streamlit Community Cloud
1. 登入 https://share.streamlit.io/
2. 選擇部署 App。
3. Repository 選擇 `polly407/yolo26n-user-site`。
4. Branch 選擇 `main`，主程式填 `app.py`。
5. 點選 Deploy。

注意：部署前必須先將 `best.pt` 加入 Repository。請勿上傳密碼、Token 或其他機密資訊。
