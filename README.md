# AI / Human 文章偵測器 (Streamlit)

單檔 Streamlit 應用，輸入文本即可判斷 AI 與 Human 機率，優先使用 Hugging Face 模型 `Hello-SimpleAI/HC3-Chinese-RoBERTa-wwm-ext`（支援中文），若模型無法載入則退回啟發式判斷。

## 快速開始

```bash
pip install streamlit transformers torch
streamlit run app.py
```

預設頁面：`http://localhost:8501`

## 功能
- 即時顯示 AI / Human 機率與進度條。
- 側欄顯示模型載入狀態與錯誤原因。
- 中文/中英文混合文本支援。
- 模型不可用時自動改用啟發式偵測。

## 注意事項
- 首次載入模型需下載權重，可能花費數秒至數十秒（視網路/GPU 而定）。
- 應用已設定 `STREAMLIT_WATCHER_TYPE=none` 以避免 torch 被檔案監看器干擾。
- 若無網路或未安裝 `transformers`/`torch`，介面仍可運作但會顯示啟發式結果。

## 檔案
- `app.py`：主程式，包含模型推論與啟發式邏輯。
- `README.md`：本說明文件。
