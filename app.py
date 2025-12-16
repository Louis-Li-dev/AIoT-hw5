"""
Streamlit app: AI vs Human text detector with multilingual support.
Uses Hugging Face text-classification models with multiple fallback options.
"""

from __future__ import annotations

import os
import sys

# CRITICAL: Set this BEFORE importing streamlit to avoid torch watcher issues
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from typing import Tuple

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception as exc:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    pipeline = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Available models (in priority order)
AVAILABLE_MODELS = {

    "fakespot-ai/roberta-base-ai-text-detection-v1": {
        "name": "Fakespot AI Detector",
        "lang": "English",
        "description": "Modern AI text detection model"
    },
}

DEFAULT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"

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


@st.cache_resource(show_spinner=True)
def load_detector(model_name: str):
    """Load the Hugging Face classifier. Falls back to None if unavailable."""
    if _IMPORT_ERROR is not None:
        return None, f"Import error: {_IMPORT_ERROR}"
    try:
        import torch
    except Exception as exc:
        return None, f"Torch import error: {exc}"
    
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


def predict_with_model(clf, text: str) -> Tuple[float, float]:
    """Return (ai_prob, human_prob) using the HF pipeline."""
    outputs = clf(text, truncation=True, max_length=512)
    
    # Handle different output formats
    if isinstance(outputs, list) and outputs:
        if isinstance(outputs[0], dict) and "label" in outputs[0]:
            # Single prediction returned as list of dicts
            outputs = [outputs]
        elif isinstance(outputs[0], list):
            # Already in correct format
            pass
    
    # Get the best prediction
    best = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
    label_str = str(best.get("label", "")).lower()
    
    # Normalize label to "ai" or "human"
    label = LABEL_TO_CLASS.get(label_str, "ai")
    score = float(best.get("score", 0.5))
    
    # Calculate AI probability
    if label == "ai":
        ai_prob = score
    else:
        ai_prob = 1 - score
    
    ai_prob = min(max(ai_prob, 0.0), 1.0)
    return ai_prob, 1 - ai_prob


def fallback_heuristic(text: str) -> Tuple[float, float]:
    """
    Lightweight heuristic: mixes length, repetition, and punctuation richness.
    Returns (ai_prob, human_prob).
    """
    stripped = text.strip()
    if not stripped:
        return 0.5, 0.5
    
    length = len(stripped)
    unique_ratio = len(set(stripped)) / max(length, 1)
    punctuation = sum(ch in "，。、！？,.?!；;：" for ch in stripped) / max(length, 1)
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(length, 1)
    
    # AI text tends to be more uniform and have less punctuation variety
    ai_score = (
        0.35 * (1 - unique_ratio)
        + 0.35 * max(0.0, 0.15 - punctuation)
        + 0.3 * min(0.2, digit_ratio) * 5
    )
    ai_prob = min(max(ai_score, 0.0), 1.0)
    return ai_prob, 1 - ai_prob


def main():
    st.set_page_config(page_title="AI / Human 文章偵測器", page_icon="🤖", layout="wide")
    st.title("🤖 AI / Human 文章偵測器")
    st.write("輸入任意文本，立即判斷是 AI 還是人類撰寫。")

    with st.sidebar:
        st.subheader("⚙️ 設定")
        
        # Model selection
        model_choice = st.selectbox(
            "選擇模型",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x]["name"],
            index=0
        )
        
        st.caption(f"語言: {AVAILABLE_MODELS[model_choice]['lang']}")
        st.caption(f"{AVAILABLE_MODELS[model_choice]['description']}")
        
        st.markdown("---")
        st.subheader("📊 模型狀態")
        
        clf, load_err = load_detector(model_choice)
        
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

    # Main content area
    default_text = "This is a sample text to test the AI/Human detector. The quick brown fox jumps over the lazy dog."
    text = st.text_area(
        "輸入文本 (建議使用英文以獲得最佳效果)",
        value=default_text,
        height=220,
        help="輸入您想要檢測的文本"
    )

    if text.strip():
        with st.spinner("分析中..."):
            if clf:
                ai_prob, human_prob = predict_with_model(clf, text)
                method_used = f"🔬 {AVAILABLE_MODELS[model_choice]['name']}"
            else:
                ai_prob, human_prob = fallback_heuristic(text)
                method_used = "📐 啟發式偵測"

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
        
    else:
        st.info("👆 請在上方輸入文本以進行偵測")


if __name__ == "__main__":
    main()
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