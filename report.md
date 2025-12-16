# AI 文本檢測器程式碼逐行解釋

## 📦 匯入模組區段（第 1-23 行）

### 第 1-5 行：檔案說明文件

```python
"""
Streamlit app: AI vs Human text detector with multilingual support.
Uses Hugging Face text-classification models with multiple fallback options.
"""
```

**說明：** 這是程式的文檔字串，描述這是一個使用 Hugging Face 模型的 AI/人類文本檢測器，支援多語言和多個備用選項。

---

### 第 7 行：引入未來特性

```python
from __future__ import annotations
```

**說明：** 啟用延遲類型註解評估，讓我們可以在類型提示中使用字串形式的類型，提高程式碼的相容性。

---

### 第 9-10 行：基礎模組

```python
import os
import sys
```

**說明：**

* `os`：用於存取作業系統功能，特別是環境變數
* `sys`：用於存取系統特定的參數和函數

---

### 第 12-13 行：關鍵修復

```python
# CRITICAL: Set this BEFORE importing streamlit to avoid torch watcher issues
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
```

**說明：****這是最重要的一行！** 在載入 Streamlit 之前，將檔案監視器類型設為 "none"，避免 Streamlit 與 PyTorch 內部結構產生衝突。這解決了你遇到的 `torch.classes` 錯誤。

---

### 第 15 行：載入 Streamlit

```python
import streamlit as st
```

**說明：** 載入 Streamlit 框架，這是建立網頁應用程式的主要工具。**必須在設定環境變數之後才能載入。**

---

### 第 16 行：型別提示

```python
from typing import Tuple
```

**說明：** 從 `typing` 模組載入 `Tuple` 類型，用於函數回傳值的型別註解。

---

### 第 18-25 行：條件式匯入 Transformers

```python
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception as exc:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    pipeline = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
```

**說明：**

* 嘗試從 `transformers` 套件載入三個關鍵元件：
  * `AutoTokenizer`：自動選擇適當的文本分詞器
  * `AutoModelForSequenceClassification`：用於序列分類的模型
  * `pipeline`：簡化模型使用的管道工具
* 如果載入失敗（例如未安裝套件），將這些變數設為 `None` 並記錄錯誤
* 這種設計讓程式即使沒有安裝 AI 模型也能運行（使用啟發式方法）

---

## 🔧 配置設定區段（第 28-50 行）

### 第 28-36 行：可用模型字典

```python
AVAILABLE_MODELS = {
    "fakespot-ai/roberta-base-ai-text-detection-v1": {
        "name": "Fakespot AI Detector",
        "lang": "English",
        "description": "Modern AI text detection model"
    },
}
```

**說明：**

* 定義一個字典，存儲可用的 AI 檢測模型
* 鍵：Hugging Face 上的模型識別碼
* 值：包含模型名稱、支援語言、描述的子字典
* 目前只有一個模型（你可以新增更多）

---

### 第 38 行：預設模型

```python
DEFAULT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
```

**說明：** 定義預設使用的模型（雖然這個模型不在 `AVAILABLE_MODELS` 中，可能是程式碼的遺留部分）。

---

### 第 40-49 行：標籤對應字典

```python
LABEL_TO_CLASS = {
    "LABEL_0": "human",
    "LABEL_1": "ai",
    "human": "human",
    "ai": "ai",
    "fake": "ai",
    "real": "human",
    "0": "human",
    "1": "ai",
}
```

**說明：**

* 建立標籤映射表，將不同模型輸出的標籤統一轉換為 "human" 或 "ai"
* 不同模型可能使用不同的標籤格式（LABEL\_0、0、human、real 等）
* 這個字典確保所有格式都能正確解讀

---

## 🧠 模型載入函數（第 52-75 行）

### 第 52-53 行：函數定義與裝飾器

```python
@st.cache_resource(show_spinner=True)
def load_detector(model_name: str):
```

**說明：**

* `@st.cache_resource`：Streamlit 裝飾器，將模型快取在記憶體中，避免重複載入
* `show_spinner=True`：載入時顯示旋轉圖示
* 函數接收 `model_name`（字串）作為參數

---

### 第 54 行：文檔字串

```python
    """Load the Hugging Face classifier. Falls back to None if unavailable."""
```

**說明：** 函數說明：載入 Hugging Face 分類器，如果無法載入則返回 None。

---

### 第 55-56 行：檢查 Transformers 匯入

```python
    if _IMPORT_ERROR is not None:
        return None, f"Import error: {_IMPORT_ERROR}"
```

**說明：** 如果之前載入 `transformers` 時發生錯誤，直接返回 None 和錯誤訊息，不嘗試載入模型。

---

### 第 57-59 行：檢查 PyTorch

```python
    try:
        import torch
    except Exception as exc:
        return None, f"Torch import error: {exc}"
```

**說明：**

* 嘗試載入 PyTorch（深度學習框架）
* 這裡使用延遲載入（lazy import），避免啟動時的問題
* 如果載入失敗，返回錯誤訊息

---

### 第 61-75 行：載入模型主體

```python
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch and torch.cuda.is_available() else -1
        clf = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
        )
        return clf, None
    except Exception as exc:
        return None, str(exc)
```

**說明：**

* **第 62 行：** 從 Hugging Face 載入分詞器
* **第 63 行：** 從 Hugging Face 載入預訓練模型
* **第 64 行：** 決定使用 GPU（device=0）或 CPU（device=-1）
  * 如果有 CUDA GPU 可用，使用 GPU 加速
* **第 65-70 行：** 建立 pipeline（管道）物件
  * `task="text-classification"`：指定任務為文本分類
  * `top_k=None`：返回所有類別的機率
* **第 71 行：** 成功時返回 pipeline 和 None（無錯誤）
* **第 73 行：** 失敗時返回 None 和錯誤訊息

---

## 🎯 模型預測函數（第 78-103 行）

### 第 78-79 行：函數定義

```python
def predict_with_model(clf, text: str) -> Tuple[float, float]:
    """Return (ai_prob, human_prob) using the HF pipeline."""
```

**說明：**

* 接收 pipeline 物件和文本字串
* 返回一個包含兩個浮點數的元組：(AI 機率, 人類機率)

---

### 第 80 行：執行推論

```python
    outputs = clf(text, truncation=True, max_length=512)
```

**說明：**

* 使用 pipeline 對文本進行分類
* `truncation=True`：如果文本太長則截斷
* `max_length=512`：最大處理 512 個 token（約 300-400 個英文單字）

---

### 第 82-89 行：處理輸出格式

```python
    # Handle different output formats
    if isinstance(outputs, list) and outputs:
        if isinstance(outputs[0], dict) and "label" in outputs[0]:
            # Single prediction returned as list of dicts
            outputs = [outputs]
        elif isinstance(outputs[0], list):
            # Already in correct format
            pass
```

**說明：**

* 不同模型和配置可能返回不同格式的輸出
* 這段程式碼統一輸出格式為 `[[{label: ..., score: ...}, ...]]`
* **第 84-86 行：** 如果是單一預測返回為字典列表，將其包裝成二維列表
* **第 87-89 行：** 如果已經是正確格式，不做任何處理

---

### 第 91-92 行：提取最佳預測

```python
    # Get the best prediction
    best = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
    label_str = str(best.get("label", "")).lower()
```

**說明：**

* 獲取信心度最高的預測結果
* 將標籤轉換為小寫字串，方便後續處理

---

### 第 94-96 行：標籤標準化

```python
    # Normalize label to "ai" or "human"
    label = LABEL_TO_CLASS.get(label_str, "ai")
    score = float(best.get("score", 0.5))
```

**說明：**

* 使用之前定義的 `LABEL_TO_CLASS` 字典將標籤轉換為 "ai" 或 "human"
* 提取信心度分數（預設 0.5 表示不確定）

---

### 第 98-103 行：計算機率

```python
    # Calculate AI probability
    if label == "ai":
        ai_prob = score
    else:
        ai_prob = 1 - score
  
    ai_prob = min(max(ai_prob, 0.0), 1.0)
    return ai_prob, 1 - ai_prob
```

**說明：**

* 如果預測為 AI，分數就是 AI 機率
* 如果預測為人類，AI 機率 = 1 - 分數
* 確保機率在 0.0 到 1.0 之間（使用 `min` 和 `max` 限制範圍）
* 返回 (AI 機率, 人類機率)

---

## 📊 啟發式備用方案（第 106-125 行）

### 第 106-110 行：函數定義與基礎檢查

```python
def fallback_heuristic(text: str) -> Tuple[float, float]:
    """
    Lightweight heuristic: mixes length, repetition, and punctuation richness.
    Returns (ai_prob, human_prob).
    """
    stripped = text.strip()
    if not stripped:
        return 0.5, 0.5
```

**說明：**

* 當模型無法載入時使用的簡單規則判斷法
* 如果文本為空，返回 50% 機率（不確定）

---

### 第 112-115 行：計算文本特徵

```python
    length = len(stripped)
    unique_ratio = len(set(stripped)) / max(length, 1)
    punctuation = sum(ch in "，。、！？,.?!；;：" for ch in stripped) / max(length, 1)
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(length, 1)
```

**說明：**

* **第 112 行：** 文本長度
* **第 113 行：** 獨特字元比例（字元種類 ÷ 總字元數）
  * AI 文本通常重複性較高，這個比例較低
* **第 114 行：** 標點符號密度（支援中英文標點）
  * AI 文本標點使用可能較單調
* **第 115 行：** 數字佔比
  * 某些 AI 生成文本可能包含較多數字

---

### 第 117-122 行：計算 AI 分數

```python
    # AI text tends to be more uniform and have less punctuation variety
    ai_score = (
        0.35 * (1 - unique_ratio)
        + 0.35 * max(0.0, 0.15 - punctuation)
        + 0.3 * min(0.2, digit_ratio) * 5
    )
```

**說明：**

* 使用加權組合計算 AI 分數：
  * **35% 權重：** 低獨特性（1 - unique\_ratio）
  * **35% 權重：** 低標點密度（標點 < 15% 時計分）
  * **30% 權重：** 數字比例（最多計算到 20%）
* 這些特徵是基於經驗觀察，實際效果有限

---

### 第 123-124 行：返回結果

```python
    ai_prob = min(max(ai_score, 0.0), 1.0)
    return ai_prob, 1 - ai_prob
```

**說明：**

* 確保 AI 機率在 0.0 到 1.0 之間
* 返回 (AI 機率, 人類機率)

---

## 🖥️ 主程式介面（第 127-228 行）

### 第 127-130 行：頁面配置

```python
def main():
    st.set_page_config(page_title="AI / Human 文章偵測器", page_icon="🤖", layout="wide")
    st.title("🤖 AI / Human 文章偵測器")
    st.write("輸入任意文本，立即判斷是 AI 還是人類撰寫。")
```

**說明：**

* 設定網頁標題、圖示和寬版面配置
* 顯示應用程式標題和說明文字

---

### 第 132-142 行：側邊欄模型選擇

```python
    with st.sidebar:
        st.subheader("⚙️ 設定")
      
        # Model selection
        model_choice = st.selectbox(
            "選擇模型",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x]["name"],
            index=0
        )
```

**說明：**

* 在側邊欄建立下拉選單讓使用者選擇模型
* `options`：模型識別碼列表
* `format_func`：顯示友善的模型名稱而非識別碼
* `index=0`：預設選擇第一個模型

---

### 第 144-145 行：顯示模型資訊

```python
        st.caption(f"語言: {AVAILABLE_MODELS[model_choice]['lang']}")
        st.caption(f"{AVAILABLE_MODELS[model_choice]['description']}")
```

**說明：** 顯示所選模型支援的語言和描述。

---

### 第 147-149 行：分隔線與狀態區塊

```python
        st.markdown("---")
        st.subheader("📊 模型狀態")
      
        clf, load_err = load_detector(model_choice)
```

**說明：**

* 新增水平分隔線
* 呼叫 `load_detector` 載入模型（由於有 `@st.cache_resource`，只會載入一次）

---

### 第 151-161 行：顯示載入狀態

```python
        if clf:
            st.success(f"✅ 模型已載入")
            st.caption(f"使用模型: {AVAILABLE_MODELS[model_choice]['name']}")
        else:
            st.warning("⚠️ 使用簡易啟發式偵測")
            if load_err:
                with st.expander("查看錯誤詳情"):
                    st.code(load_err, language="text")
            elif _IMPORT_ERROR:
                with st.expander("查看匯入錯誤"):
                    st.code(str(_IMPORT_ERROR), language="text")
```

**說明：**

* 如果模型載入成功，顯示綠色成功訊息
* 如果載入失敗，顯示黃色警告並提供可展開的錯誤詳情
* 讓使用者了解當前使用的是模型還是啟發式方法

---

### 第 163-178 行：使用提示

```python
        st.markdown("---")
        st.subheader("💡 使用提示")
        st.markdown("""
        **安裝依賴：**
        ```bash
        pip install streamlit transformers torch
        ```
      
        **注意事項：**
        - 英文文本效果最佳
        - 中文需使用啟發式方法
        - 首次載入需下載模型（約 500MB）
        - GPU 可加速推論
        """)
```

**說明：**

* 在側邊欄顯示安裝說明和使用注意事項
* 使用 Markdown 格式化文字，包含程式碼區塊

---

### 第 180-187 行：文本輸入區

```python
    # Main content area
    default_text = "This is a sample text to test the AI/Human detector. The quick brown fox jumps over the lazy dog."
    text = st.text_area(
        "輸入文本 (建議使用英文以獲得最佳效果)",
        value=default_text,
        height=220,
        help="輸入您想要檢測的文本"
    )
```

**說明：**

* 建立多行文本輸入框
* 提供預設範例文本
* 高度設為 220 像素
* 提供提示訊息

---

### 第 189-196 行：執行檢測

```python
    if text.strip():
        with st.spinner("分析中..."):
            if clf:
                ai_prob, human_prob = predict_with_model(clf, text)
                method_used = f"🔬 {AVAILABLE_MODELS[model_choice]['name']}"
            else:
                ai_prob, human_prob = fallback_heuristic(text)
                method_used = "📐 啟發式偵測"
```

**說明：**

* 只在有輸入文本時才執行檢測
* 顯示旋轉載入動畫
* 如果有模型，使用模型預測；否則使用啟發式方法
* 記錄使用的方法以便顯示

---

### 第 198-212 行：顯示結果

```python
        # Display results
        st.markdown("### 📈 檢測結果")
      
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "🤖 AI 機率",
                f"{ai_prob * 100:.1f}%",
                delta=f"{(ai_prob - 0.5) * 100:+.1f}%" if ai_prob != 0.5 else None
            )
            st.progress(min(ai_prob, 1.0))
          
        with col2:
            st.metric(
                "👤 Human 機率",
                f"{human_prob * 100:.1f}%",
                delta=f"{(human_prob - 0.5) * 100:+.1f}%" if human_prob != 0.5 else None
            )
            st.progress(min(human_prob, 1.0))
```

**說明：**

* 建立兩欄布局顯示結果
* **左欄：** AI 機率，使用 `st.metric` 顯示數值和與 50% 的差異
* **右欄：** 人類機率
* 兩欄都包含進度條視覺化
* `delta` 參數顯示偏離 50% 的程度（正值顯示綠色上箭頭，負值顯示紅色下箭頭）

---

### 第 214-225 行：結果詮釋

```python
        # Interpretation
        st.markdown("---")
        if ai_prob > 0.75:
            st.error("🤖 **很可能是 AI 生成的文本**")
        elif ai_prob > 0.6:
            st.warning("⚠️ **可能是 AI 生成的文本**")
        elif ai_prob > 0.4:
            st.info("🤔 **無法確定來源**")
        elif ai_prob > 0.25:
            st.warning("⚠️ **可能是人類撰寫**")
        else:
            st.success("👤 **很可能是人類撰寫**")
      
        st.caption(f"推斷方式：{method_used}")
        st.caption(f"文本長度：{len(text)} 字元")
```

**說明：**

* 根據 AI 機率給出解讀：
  * > 75%：很可能是 AI（紅色錯誤訊息）
    >
  * 60-75%：可能是 AI（黃色警告）
  * 40-60%：不確定（藍色資訊）
  * 25-40%：可能是人類（黃色警告）
  * < 25%：很可能是人類（綠色成功訊息）
* 顯示使用的檢測方法和文本長度

---

### 第 227-228 行：空文本提示

```python
    else:
        st.info("👆 請在上方輸入文本以進行偵測")
```

**說明：** 如果沒有輸入文本，顯示提示訊息。

---

## 🚀 程式進入點（第 231-232 行）

### 第 231-232 行：執行主程式

```python
if __name__ == "__main__":
    main()
```

**說明：**

* 這是 Python 的標準進入點寫法
* 當直接執行此檔案時，呼叫 `main()` 函數啟動應用程式
* 如果此檔案被其他程式匯入，則不會自動執行

---

### 第 233-245 行：註解掉的獨立測試程式碼

```python
    # if "streamlit" in sys.argv[0]:
    #     main()
    # else:
    #     # Allow running as a script for quick checks
    #     sample = "The weather is nice today, let's go for a walk in the park."
    #     print(f"Sample input: {sample}")
    #     clf, err = load_detector(DEFAULT_MODEL)
    #     if clf:
    #         ai_prob, human_prob = predict_with_model(clf, sample)
    #         print(f"AI: {ai_prob:.2%}, Human: {human_prob:.2%} (model)")
    #     else:
    #         ai_prob, human_prob = fallback_heuristic(sample)
    #         print(f"AI: {ai_prob:.2%}, Human: {human_prob:.2%} (heuristic)")
    #         if err:
    #             print(f"Error: {err}")
```

**說明：**

* 這是被註解掉的替代進入點邏輯
* 原本設計可以：
  * 如果用 Streamlit 執行，啟動網頁介面
  * 如果直接用 Python 執行，在終端機顯示範例測試結果
* 目前被註解掉，改為直接執行 `main()`

---

## 🎓 總結

這個程式的核心架構：

1. **環境設定** → 避免 PyTorch 與 Streamlit 衝突
2. **模型載入** → 從 Hugging Face 下載並快取 AI 檢測模型
3. **預測功能** → 使用模型或啟發式方法分析文本
4. **網頁介面** → 用 Streamlit 建立互動式應用程式
5. **結果視覺化** → 清楚顯示 AI/人類機率和詮釋

關鍵技術點：

* ✅ 使用 `os.environ` 預先解決衝突
* ✅ 條件式匯入確保程式總能運行
* ✅ 快取機制避免重複載入大型模型
* ✅ 多種輸出格式的處理
* ✅ 優雅的錯誤處理和使用者提示
