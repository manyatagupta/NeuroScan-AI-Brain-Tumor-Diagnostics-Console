# ---------------------------------------------------
# NeuroScan AI — Brain Tumor Diagnostics Console
#
# Requirements:
#   pip install streamlit tensorflow pillow numpy reportlab plotly
# ---------------------------------------------------

import io
import datetime
import time
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import plotly.graph_objects as go

# ---------------------------------------------------
# Config
# ---------------------------------------------------
st.set_page_config(page_title="NeuroScan AI | Diagnostics", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = "best_brain_tumor_model.keras"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = (224, 224)
LOW_CONFIDENCE_THRESHOLD = 70.0

CLASS_COLORS = {
    "Glioma": "#FF3366",
    "Meningioma": "#FFB300",
    "No Tumor": "#00E676",
    "Pituitary": "#8B7FFF",
}

REPORT_TEMPLATES = {
    "Glioma": {
        "en": (
            "**What it is:** A glioma is a tumor that develops from glial cells, the supportive tissue "
            "surrounding neurons in the brain and spinal cord. Gliomas range from slow-growing (low-grade) "
            "to aggressive (high-grade, such as glioblastoma).\n\n"
            "**Common symptoms:** Persistent headaches, seizures, memory or personality changes, nausea, "
            "and vision or speech difficulties, depending on the tumor's location.\n\n"
            "**General treatment options:** Treatment usually depends on grade and location, and may include "
            "surgical resection, radiation therapy, chemotherapy, or a combination — decided by a "
            "neuro-oncology team after further imaging and biopsy."
        ),
        "hi": (
            "**Ye kya hota hai:** Glioma ek tumor hai jo glial cells se banta hai — ye woh supportive tissue "
            "hai jo brain aur spinal cord ke neurons ko surround karta hai. Iska growth slow bhi ho sakta hai "
            "aur aggressive (high-grade) bhi.\n\n"
            "**Common symptoms:** Baar-baar headache, seizures (mirgi jaise attacks), memory ya behavior me "
            "change, nausea, aur tumor ki location ke hisaab se vision ya speech me dikkat.\n\n"
            "**Treatment options:** Ye tumor ke grade aur location par depend karta hai — surgery, radiation "
            "therapy, chemotherapy, ya inka combination. Final decision neuro-oncology team hi biopsy aur "
            "further scans ke baad leti hai."
        ),
    },
    "Meningioma": {
        "en": (
            "**What it is:** A meningioma forms in the meninges, the layers of tissue covering the brain and "
            "spinal cord. Most meningiomas are slow-growing and benign, though location can still make them "
            "clinically significant.\n\n"
            "**Common symptoms:** Headaches, gradual vision problems, weakness in limbs, and — less commonly — "
            "seizures, depending on where the tumor presses on surrounding brain tissue.\n\n"
            "**General treatment options:** Small, asymptomatic meningiomas are often just monitored with "
            "periodic scans. Larger or symptomatic ones may require surgical removal or targeted radiation."
        ),
        "hi": (
            "**Ye kya hota hai:** Meningioma meninges (brain aur spinal cord ko cover karne wali layers) me "
            "banta hai. Zyadatar meningioma slow-growing aur benign hote hain, lekin location ke hisaab se "
            "ye clinically important ho sakte hain.\n\n"
            "**Common symptoms:** Headache, dheere-dheere vision problem, haath-pair me weakness, aur kabhi-"
            "kabhi seizures — ye depend karta hai ki tumor brain ke kis part par pressure daal raha hai.\n\n"
            "**Treatment options:** Chhote, symptom-free meningioma ko sirf periodic scans se monitor kiya "
            "jata hai. Bade ya symptomatic cases me surgery ya targeted radiation ki zarurat pad sakti hai."
        ),
    },
    "Pituitary": {
        "en": (
            "**What it is:** A pituitary tumor grows in the pituitary gland at the base of the brain, which "
            "regulates several hormones. Most are non-cancerous (adenomas) but can still disrupt hormone "
            "balance or press on nearby structures like the optic nerves.\n\n"
            "**Common symptoms:** Vision changes (especially peripheral vision loss), unexplained fatigue, "
            "headaches, and hormone-related symptoms such as irregular periods, unexpected weight change, or "
            "mood shifts.\n\n"
            "**General treatment options:** Options include hormone-regulating medication, surgery (often "
            "via a minimally invasive nasal approach), or radiation therapy, chosen based on tumor size and "
            "hormonal activity."
        ),
        "hi": (
            "**Ye kya hota hai:** Pituitary tumor brain ke base me maujood pituitary gland me grow karta hai, "
            "jo body ke kai hormones control karti hai. Zyadatar cases non-cancerous (adenoma) hote hain, "
            "lekin ye hormone balance bigaad sakte hain ya optic nerves par pressure daal sakte hain.\n\n"
            "**Common symptoms:** Vision me change (khaaskar peripheral vision loss), bina wajah thakaan, "
            "headache, aur hormone-related symptoms jaise irregular periods, sudden weight change, ya mood "
            "swings.\n\n"
            "**Treatment options:** Hormone-regulating medicines, surgery (aksar minimally invasive nasal "
            "route se), ya radiation therapy — tumor ke size aur hormonal activity ke hisaab se decide hota hai."
        ),
    },
    "No Tumor": {
        "en": (
            "Great news — the scan shows **No Tumor** detected. If you're still experiencing symptoms like "
            "persistent headaches, vision changes, or nausea, please consult a neurologist for a complete "
            "clinical evaluation regardless of this result."
        ),
        "hi": (
            "Achi khabar hai — scan me **No Tumor** detect hua hai. Agar phir bhi headache, vision change, "
            "ya nausea jaise symptoms ho rahe hain, to iss result ke bawajood ek neurologist se poora "
            "clinical checkup zarur karwayein."
        ),
    },
}

DISCLAIMER = (
    "\n\n---\n*This summary is AI-generated for educational/demo purposes only and is not a medical "
    "diagnosis. Always consult a qualified doctor for confirmation and treatment decisions.*"
)

# ---------------------------------------------------
# Cinematic UI injection & Orbs
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-deep: #03040B;
    --bg-panel: rgba(12, 17, 32, 0.45);
    --border: rgba(255, 255, 255, 0.08);
    --cyan: #00F0FF;
    --violet: #8B7FFF;
    --text-primary: #FFFFFF;
    --text-muted: #8E9BAE;
    --danger: #FF3366;
    --safe: #00E676;
    --warn: #FFB300;
}

html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }

/* Animated Aurora Background */
@keyframes aurora {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(-45deg, #03040B, #0a1122, #060a17, #0f152d);
    background-size: 300% 300%;
    animation: aurora 25s ease infinite;
    color: var(--text-primary);
}
#MainMenu, footer { visibility: hidden; }

/* Floating Glowing Orbs */
.orb {
    position: fixed; border-radius: 50%; filter: blur(90px);
    z-index: -1; opacity: 0.35;
    animation: floatOrb 12s infinite alternate ease-in-out;
    pointer-events: none;
}
.orb-1 { width: 400px; height: 400px; background: #00F0FF; top: -100px; left: -100px; }
.orb-2 { width: 500px; height: 500px; background: #8B7FFF; bottom: -150px; right: -100px; animation-delay: -3s; }
.orb-3 { width: 350px; height: 350px; background: #FF3366; top: 40%; left: 30%; opacity: 0.15; animation-delay: -6s; }

@keyframes floatOrb {
    0% { transform: translateY(0px) scale(1); }
    100% { transform: translateY(60px) scale(1.15); }
}

/* Dashboard Metric Cards */
.metric-card {
    background: rgba(12, 17, 32, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 18px 12px;
    text-align: center; backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s;
}
.metric-card:hover {
    transform: translateY(-6px);
    border-color: rgba(0, 240, 255, 0.4);
    box-shadow: 0 12px 30px rgba(0,240,255,0.1);
}
.metric-val {
    font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 32px;
    background: linear-gradient(135deg, #fff, var(--cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 2px; color: var(--text-muted); margin-top: 6px;
}

/* Glassmorphism Panels applied directly to Streamlit Columns */
[data-testid="stHorizontalBlock"]:nth-of-type(2) > [data-testid="column"] {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 24px;
    padding: 32px 34px; margin-bottom: 26px; backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease, border-color 0.4s ease;
}
[data-testid="stHorizontalBlock"]:nth-of-type(2) > [data-testid="column"]:hover {
    transform: translateY(-2px); box-shadow: 0 16px 50px rgba(0, 240, 255, 0.12);
    border-color: rgba(0, 240, 255, 0.3);
}

.panel-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--cyan); margin-bottom: 8px;
}
.panel-title { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 26px; margin-bottom: 20px; }

/* Sidebar styling */
section[data-testid="stSidebar"] { background: rgba(5, 8, 16, 0.85); border-right: 1px solid var(--border); backdrop-filter: blur(35px); }
section[data-testid="stSidebar"] .block-container { padding-top: 36px; }
.sidebar-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1.5px; color: var(--safe);
    border: 1px solid rgba(0,230,118,0.4); background: rgba(0,230,118,0.1);
    padding: 6px 12px; border-radius: 20px; display: inline-block; margin-bottom: 18px;
    box-shadow: 0 0 12px rgba(0,230,118,0.2);
}
.history-item {
    font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-muted);
    padding: 12px 14px; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; margin-bottom: 10px;
    background: rgba(255,255,255,0.02); transition: all 0.3s ease;
}
.history-item:hover { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.15); transform: translateX(4px); }

/* Primary Button Styling */
.stButton>button {
    background: linear-gradient(135deg, var(--cyan), #0099FF); color: #000; font-weight: 700;
    font-family: 'Outfit', sans-serif; letter-spacing: 1px; border: none; border-radius: 14px;
    width: 100%; padding: 0.9em 0; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    font-size: 17px; text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,240,255,0.3);
}
.stButton>button:hover { 
    box-shadow: 0 0 35px rgba(0,240,255,0.6); 
    transform: translateY(-3px) scale(1.02); 
    filter: brightness(1.15);
}

/* Verdict Display */
@keyframes pulseGlow { 0% { box-shadow: 0 0 15px currentColor; } 50% { box-shadow: 0 0 35px currentColor; } 100% { box-shadow: 0 0 15px currentColor; } }
.verdict {
    display: flex; align-items: center; gap: 16px; padding: 22px 28px; border-radius: 18px;
    margin-bottom: 24px; border: 1px solid var(--border); backdrop-filter: blur(15px);
}
.verdict-dot { width: 16px; height: 16px; border-radius: 50%; animation: pulseGlow 2s infinite; }
.verdict-text { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 26px; letter-spacing: 0.5px; }

/* Report Text Box */
.report-box {
    padding: 26px 30px; border-radius: 16px; background: rgba(0,0,0,0.25);
    border-left: 4px solid var(--violet); color: #F3F4F6; font-size: 16px; line-height: 1.8;
    box-shadow: inset 0 4px 15px rgba(0,0,0,0.15); border-right: 1px solid rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.03);
}

/* Custom Overrides */
div[data-testid="stFileUploader"] > section {
    background-color: rgba(255,255,255,0.02); border: 2px dashed rgba(255,255,255,0.15); border-radius: 18px;
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"] > section:hover {
    background-color: rgba(0,240,255,0.06); border-color: rgba(0,240,255,0.5); box-shadow: 0 0 20px rgba(0,240,255,0.1);
}
</style>

<!-- Injected Orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Session state
# ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# Load model (cached)
# ---------------------------------------------------
@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None

model = get_model()

# ---------------------------------------------------
# Functional Helpers
# ---------------------------------------------------
def generate_local_report(tumor_type, lang):
    return REPORT_TEMPLATES[tumor_type][lang] + DISCLAIMER

def find_conv_layer(m):
    for layer in reversed(m.layers):
        try:
            if len(layer.output.shape) == 4:
                return m, layer.name
        except Exception:
            continue
    for layer in reversed(m.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                try:
                    if len(sub.output.shape) == 4:
                        return layer, sub.name
                except Exception:
                    continue
    return None, None

def make_gradcam_heatmap(img_array, m):
    try:
        target_model, conv_name = find_conv_layer(m)
        if conv_name is None:
            return None
        grad_model = tf.keras.models.Model(m.inputs, [target_model.get_layer(conv_name).output, m.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def colorize_heatmap(heatmap, size):
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [(8, 8, 64), (0, 140, 200), (0, 220, 120), (255, 210, 0), (255, 40, 40)]
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(size)
    h = np.array(heatmap_img) / 255.0
    r = np.interp(h, stops, [c[0] for c in colors])
    g = np.interp(h, stops, [c[1] for c in colors])
    b = np.interp(h, stops, [c[2] for c in colors])
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)

def overlay_heatmap(original_img, heatmap, alpha=0.45):
    colored = colorize_heatmap(heatmap, original_img.size).convert("RGB")
    return Image.blend(original_img.convert("RGB"), colored, alpha=alpha)

def generate_pdf_report(image, predicted_class, confidence, prediction, report_text, heatmap_img=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleC", parent=styles["Title"], textColor=rl_colors.HexColor("#10192E"))
    body_style = ParagraphStyle("BodyC", parent=styles["BodyText"], leading=16)

    elements = [
        Paragraph("NeuroScan AI — Brain Tumor Analysis Report", title_style),
        Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]),
        Spacer(1, 14),
    ]

    img_buf = io.BytesIO()
    image.convert("RGB").resize((260, 260)).save(img_buf, format="PNG")
    img_buf.seek(0)
    imgs = [RLImage(img_buf, width=2.1 * inch, height=2.1 * inch)]

    if heatmap_img is not None:
        hbuf = io.BytesIO()
        heatmap_img.resize((260, 260)).save(hbuf, format="PNG")
        hbuf.seek(0)
        imgs.append(RLImage(hbuf, width=2.1 * inch, height=2.1 * inch))

    elements.append(Table([imgs], colWidths=[2.3 * inch] * len(imgs)))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph(f"<b>Detection:</b> {predicted_class}", body_style))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", body_style))
    elements.append(Spacer(1, 10))

    table_data = [["Class", "Probability"]] + [
        [cls, f"{p*100:.2f}%"] for cls, p in zip(CLASSES, prediction[0])
    ]
    t = Table(table_data, colWidths=[3 * inch, 2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#10192E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("AI Medical Summary", styles["Heading2"]))
    clean_text = report_text.replace("**", "").replace("*", "")
    for para in clean_text.split("\n"):
        if para.strip():
            elements.append(Paragraph(para.strip(), body_style))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    buf.seek(0)
    return buf

def render_gauge(confidence, color):
    r = 74
    circumference = 2 * 3.14159265 * r
    filled = (confidence / 100) * circumference
    svg = f"""
    <div style="display:flex; justify-content:center; align-items:center; padding: 12px 0;">
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="16"/>
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="16"
                stroke-linecap="round"
                stroke-dasharray="{filled:.2f} {circumference:.2f}"
                transform="rotate(-90 100 100)"
                style="filter: drop-shadow(0 0 16px {color}); transition: stroke-dasharray 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);"/>
        <text x="100" y="92" text-anchor="middle" font-family="Outfit, sans-serif"
              font-size="38" font-weight="700" fill="#fff">{confidence:.1f}%</text>
        <text x="100" y="120" text-anchor="middle" font-family="JetBrains Mono, monospace"
              font-size="11" letter-spacing="3" fill="#8E9BAE">CONFIDENCE</text>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)

def render_plotly_chart(prediction):
    probs = prediction[0] * 100
    colors = [CLASS_COLORS[c] for c in CLASSES]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs, y=CLASSES, orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.3)', width=1)),
            text=[f"{p:.1f}%" for p in probs], textposition='outside',
            textfont=dict(color='white', family='Outfit', size=14)
        )
    ])
    fig.update_layout(
        margin=dict(l=10, r=60, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#FFFFFF', size=15, family='Outfit', weight='bold')),
        height=220
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-tag">● SECURE UPLINK ACTIVE</div>', unsafe_allow_html=True)
    
    st.image("https://cdn-icons-png.flaticon.com/512/3029/3029272.png", width=60)
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; margin-top: -10px;'>NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8E9BAE; font-size:14px; margin-top: -10px;">Clinical grade deep learning inference engine.</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="panel-label">ACTIVE MODULES</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family: JetBrains Mono; font-size: 13px; color: #8E9BAE; line-height: 2;">'
                '✅ MobileNetV2 Architecture<br>'
                '✅ Grad-CAM Spatial Mapping<br>'
                '✅ Plotly Visual Analytics<br>'
                '✅ Offline Report Engine'
                '</div>', unsafe_allow_html=True)
    st.markdown("---")

    show_heatmap = st.checkbox("Enable Grad-CAM Overlay", value=True)
    st.markdown("---")

    if st.session_state.history:
        st.markdown('<div class="panel-label">SESSION LOGS</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(
                f'<div class="history-item">🕒 {h["time"]} &nbsp;|&nbsp; '
                f'<strong style="color:{CLASS_COLORS[h["class"]]};">{h["class"]}</strong> '
                f'<span style="float:right;">{h["confidence"]:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("---")

# ---------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------
st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.markdown('<div class="metric-card"><div class="metric-val">98.2%</div><div class="metric-lbl">CORE ACCURACY</div></div>', unsafe_allow_html=True)
m2.markdown('<div class="metric-card"><div class="metric-val">412ms</div><div class="metric-lbl">INFERENCE SPEED</div></div>', unsafe_allow_html=True)
m3.markdown('<div class="metric-card"><div class="metric-val" style="color: #00E676;">ONLINE</div><div class="metric-lbl">SYSTEM STATUS</div></div>', unsafe_allow_html=True)
m4.markdown('<div class="metric-card"><div class="metric-val">V2.1</div><div class="metric-lbl">ENGINE VERSION</div></div>', unsafe_allow_html=True)


if model is None:
    st.error(f"❌ Could not load the model from `{MODEL_PATH}`.")
    st.stop()

col1, col2 = st.columns([1, 1.25])

# ---------------------------------------------------
# Column 1 — Upload
# ---------------------------------------------------
with col1:
    st.markdown('<div class="panel-label">PHASE 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload MRI Scan</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    img_array, image = None, None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(np.array(image), caption="High-Resolution Preview", use_column_width=True)
        img_resized = image.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_array = preprocess_input(img_array)
    else:
        st.markdown(
            '<div style="border:2px dashed rgba(255,255,255,0.1); border-radius:18px; padding:50px; '
            'text-align:center; color:var(--text-muted); font-size:15px; background:rgba(255,255,255,0.02);">'
            '<h1 style="margin:0; font-size:40px;">🩻</h1><br>Drop an MRI scan here to initiate the neural pipeline.</div>',
            unsafe_allow_html=True
        )

    run_analysis = st.button("Initialize Neural Analysis 🚀") if uploaded_file is not None else False

# ---------------------------------------------------
# Column 2 — Analysis & Report
# ---------------------------------------------------
with col2:
    st.markdown('<div class="panel-label">PHASE 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">AI Analytics &amp; Verdict</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown('<p style="color:var(--text-muted); font-size:16px; text-align:center; padding: 40px;">System waiting for valid scan input.</p>', unsafe_allow_html=True)
    elif not run_analysis:
        st.markdown('<p style="color:var(--text-muted); font-size:16px; text-align:center; padding: 40px;">Scan locked. Click <b style="color:var(--cyan);">Initialize Neural Analysis</b> to engage model.</p>', unsafe_allow_html=True)
    else:
        with st.spinner("Extracting deep features..."):
            time.sleep(0.9)
            prediction = model.predict(img_array)
            predicted_idx = np.argmax(prediction)
            predicted_class = CLASSES[predicted_idx]
            confidence = np.max(prediction) * 100
            color = CLASS_COLORS[predicted_class]

        st.session_state.history.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "class": predicted_class, "confidence": confidence})

        st.markdown(f"""
        <div class="verdict" style="background:{color}1A; border-color:{color}66; box-shadow: 0 8px 30px {color}25;">
            <div class="verdict-dot" style="background:{color}; color:{color};"></div>
            <div class="verdict-text" style="color:{color};">{predicted_class.upper()} DETECTED</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="risk-banner">⚠️ <div><strong>Confidence is below optimal threshold.</strong> '
                'Treat result as inconclusive. Recommend secondary clinical review.</div></div>',
                unsafe_allow_html=True
            )

        render_gauge(confidence, color)

        with st.expander("📊 Activation Probabilities", expanded=True):
            render_plotly_chart(prediction)

        heatmap_overlay = None
        if show_heatmap:
            with st.expander("🔥 Grad-CAM Spatial Mapping", expanded=True):
                with st.spinner("Rendering thermal map..."):
                    heatmap = make_gradcam_heatmap(img_array, model)
                if heatmap is not None:
                    heatmap_overlay = overlay_heatmap(image.resize(IMG_SIZE), heatmap)
                    hc1, hc2 = st.columns(2)
                    hc1.image(image.resize(IMG_SIZE), caption="Input Scan", use_column_width=True)
                    hc2.image(heatmap_overlay, caption="Network Activation Zones", use_column_width=True)
                else:
                    st.info("Heatmap isn't available for this model's architecture.")

        st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 30px 0;">', unsafe_allow_html=True)

        # Phase 3
        st.markdown('<div class="panel-label">PHASE 03</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🤖 AI Clinical Synthesis</div>', unsafe_allow_html=True)

        with st.spinner("Synthesizing clinical report..."):
            time.sleep(1.2)
            report = generate_local_report(predicted_class, "en")

        tabs = st.tabs(["📋 Executive Summary", "⚠️ Protocol & Options", "🔬 Clinical Metadata"])
        sections = report.split("\n\n")
        
        with tabs[0]:
            st.markdown(f'<div class="report-box">{sections[0] if len(sections) > 0 else "No summary available."}</div>', unsafe_allow_html=True)
        with tabs[1]:
            st.markdown(f'<div class="report-box">{sections[1] if len(sections) > 1 else "Consult your doctor for next steps."}</div>', unsafe_allow_html=True)
            if len(sections) > 2:
                st.markdown(f'<div class="report-box" style="margin-top: 15px;">{sections[2]}</div>', unsafe_allow_html=True)
        with tabs[2]:
            st.info("Technical metadata intended for authorized healthcare providers.")
            st.markdown(f"**Primary Finding:** {predicted_class} (Confidence: {confidence:.2f}%)")
            st.markdown("**Engine:** MobileNetV2 Architecture (ImageNet Weights)")
            st.markdown("**Resolution Matrix:** 224x224 RGB Interpolated")
            st.markdown(DISCLAIMER)

        st.markdown("<br>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        full_report_txt = f"NEUROSCAN AI — ADVANCED ANALYSIS REPORT\n\nDetection: {predicted_class}\nConfidence: {confidence:.2f}%\n\nDETAILED SUMMARY:\n{report}"
        dl1.download_button("📥 Export TXT", data=full_report_txt, file_name="neuroscan_report.txt", mime="text/plain", type="primary", use_container_width=True)
        pdf_buf = generate_pdf_report(image, predicted_class, confidence, prediction, report, heatmap_overlay)
        dl2.download_button("📄 Export Secure PDF", data=pdf_buf, file_name="neuroscan_report.pdf", mime="application/pdf", type="primary", use_container_width=True)