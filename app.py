import streamlit as st
import time
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from translate import Translator

st.set_page_config(
    page_title="Google Translate (MoE 150M Transformer)",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Google Translate CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .gt-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .gt-logo {
        font-size: 26px;
        font-weight: 500;
        color: #4285F4;
    }
    .gt-sublogo {
        font-size: 22px;
        font-weight: 400;
        color: #5f6368;
    }
    .badge-moe {
        background-color: #E8F0FE;
        color: #1967D2;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        margin-left: 8px;
    }
    .translate-card {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #dadce0;
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        height: 100%;
        min-height: 240px;
    }
    .stat-pill {
        display: inline-block;
        background: #f1f3f4;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        color: #3c4043;
        margin-right: 8px;
        margin-top: 8px;
    }
    .stTextArea textarea {
        font-size: 16px;
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        resize: vertical;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="gt-header">
    <span class="gt-logo">Google</span>
    <span class="gt-sublogo">Translate</span>
    <span class="badge-moe">150M MoE (5 Experts • Top-2)</span>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.subheader("⚙️ Model & Generation Controls")
    
    ckpt_option = st.selectbox(
        "Checkpoint",
        options=[
            Config.CHECKPOINT_BEST,
            Config.CHECKPOINT_LATEST,
            "Base (Random Weights)"
        ],
        index=0
    )
    ckpt_path = None if ckpt_option == "Base (Random Weights)" else ckpt_option

    use_int8 = st.toggle("Enable INT8 Quantization (bitsandbytes)", value=False, help="Runs the MoE model in 8-bit precision on RTX 4050")

    st.markdown("---")
    st.markdown("### 🎛️ Decoding Hyperparameters")
    decoding_method = st.radio("Decoding Method", ["sampling", "greedy", "beam_search"], index=0,
                               format_func=lambda x: "Sampling (Top-k & Top-p)" if x == "sampling" else ("Greedy Search" if x == "greedy" else "Beam Search"))

    if decoding_method == "sampling":
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=Config.DEFAULT_TEMPERATURE, step=0.05)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=Config.DEFAULT_TOP_K, step=1)
        top_p = st.slider("Top-P / Top-Q (Nucleus)", min_value=0.1, max_value=1.0, value=Config.DEFAULT_TOP_P, step=0.05)
        beam_size = 1
    elif decoding_method == "beam_search":
        beam_size = st.slider("Beam Size", min_value=2, max_value=8, value=Config.DEFAULT_BEAM_SIZE, step=1)
        temperature = 1.0
        top_k = 0
        top_p = 1.0
    else:
        temperature = 0.0
        top_k = 0
        top_p = 1.0
        beam_size = 1

    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=Config.DEFAULT_REPETITION_PENALTY, step=0.05)
    max_new_tokens = st.slider("Max New Tokens", min_value=16, max_value=256, value=Config.MAX_NEW_TOKENS, step=16)

    st.markdown("---")
    st.markdown("### 🖥️ Hardware Info")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.text(f"Device: {gpu_name}")
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
        mem_res = torch.cuda.memory_reserved(0) / (1024 * 1024)
        st.text(f"VRAM: {mem_alloc:.1f} MB allocated / {mem_res:.1f} MB reserved")

# Cache Translator instance
@st.cache_resource(show_spinner="Loading 150M MoE Transformer into memory...")
def load_translator(ckpt, int8_mode):
    return Translator(model_path=ckpt, quantized_int8=int8_mode)

try:
    translator = load_translator(ckpt_path, use_int8)
except Exception as e:
    st.error(f"Error initializing translator: {e}")
    translator = None

# Language Selector State
if "src_lang" not in st.session_state:
    st.session_state.src_lang = "English"
if "tgt_lang" not in st.session_state:
    st.session_state.tgt_lang = "Vietnamese"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def swap_languages():
    st.session_state.src_lang, st.session_state.tgt_lang = st.session_state.tgt_lang, st.session_state.src_lang

# Controls row
col_ctrl1, col_swap, col_ctrl2 = st.columns([5, 1, 5])

with col_ctrl1:
    src_choice = st.selectbox("Source Language", ["English", "Vietnamese"],
                              index=0 if st.session_state.src_lang == "English" else 1, key="src_select")
    st.session_state.src_lang = src_choice

with col_swap:
    st.write("")
    st.write("")
    if st.button("⇄", help="Swap languages", use_container_width=True):
        swap_languages()
        st.rerun()

with col_ctrl2:
    tgt_choice = st.selectbox("Target Language", ["Vietnamese", "English"],
                              index=0 if st.session_state.tgt_lang == "Vietnamese" else 1, key="tgt_select")
    st.session_state.tgt_lang = tgt_choice

# Sample sentences buttons
st.markdown("**Quick Samples:**")
sample_cols = st.columns(4)
sample_prompts = [
    ("School & Learning", "Students are studying hard at school for the national exam.", "English", "Vietnamese"),
    ("Vietnamese Compound", "Trường học và thành phố là hai nơi gắn liền với tuổi trẻ.", "Vietnamese", "English"),
    ("Technology & AI", "Mixture of Experts architecture allows scaling to 150 million parameters efficiently.", "English", "Vietnamese"),
    ("Daily Greeting", "Xin chào! Rất vui được gặp bạn ngày hôm nay.", "Vietnamese", "English")
]

for idx, (label, text, s_lang, t_lang) in enumerate(sample_prompts):
    if sample_cols[idx].button(f"💡 {label}", use_container_width=True):
        st.session_state.input_text = text
        st.session_state.src_lang = s_lang
        st.session_state.tgt_lang = t_lang
        st.rerun()

# Translation Interface Panels
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Source Text")
    input_text = st.text_area(
        label="Source Input",
        value=st.session_state.input_text,
        height=180,
        placeholder="Type or paste text to translate here...",
        label_visibility="collapsed"
    )
    st.session_state.input_text = input_text

    c1, c2 = st.columns([6, 2])
    with c1:
        st.caption(f"{len(input_text)} characters")
    with c2:
        if st.button("Clear Text", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

translation_result = ""
stats_result = None

with col_right:
    st.markdown("##### Translation")
    output_container = st.empty()

    if input_text.strip() and translator:
        s_code = "en" if st.session_state.src_lang == "English" else "vi"
        t_code = "vi" if st.session_state.tgt_lang == "Vietnamese" else "en"

        with st.spinner("Translating..."):
            translation_result, stats_result = translator.translate(
                text=input_text,
                src_lang=s_code,
                tgt_lang=t_code,
                method=decoding_method,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                beam_size=beam_size,
                max_new_tokens=max_new_tokens
            )

        output_container.text_area(
            label="Translation Output",
            value=translation_result,
            height=180,
            label_visibility="collapsed"
        )
    else:
        output_container.text_area(
            label="Translation Output",
            value="",
            height=180,
            placeholder="Translation will appear here...",
            label_visibility="collapsed"
        )

# Telemetry and MoE Visualization
if stats_result:
    st.markdown("---")
    st.markdown("### 📊 Inference Metrics & MoE Expert Routing")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Latency", f"{stats_result['latency_ms']:.1f} ms")
    m_col2.metric("Tokens Generated", f"{stats_result['generated_tokens']}")
    m_col3.metric("Throughput", f"{stats_result['tokens_per_sec']:.1f} tok/s")
    m_col4.metric("Active Experts / Total", f"2 / 5 (Sparse Top-2)")

    st.markdown("#### 🧠 MoE Expert Utilization (5 Experts • Top-2 Dispatched per Token)")
    expert_cols = st.columns(5)
    
    # Inspect expert distribution from model's encoder last layer
    try:
        last_moe = translator.model.encoders[-1].moe
        counts = last_moe.last_expert_counts.cpu().tolist()
        total_c = max(1.0, sum(counts))
        for e_i in range(5):
            pct = (counts[e_i] / total_c) * 100
            expert_cols[e_i].markdown(f"**Expert {e_i + 1}**")
            expert_cols[e_i].progress(min(1.0, pct / 100.0))
            expert_cols[e_i].caption(f"{pct:.1f}% load")
    except Exception:
        for e_i in range(5):
            expert_cols[e_i].markdown(f"**Expert {e_i + 1}**")
            expert_cols[e_i].progress(0.2)
            expert_cols[e_i].caption("Active")
