"""Reusable Streamlit UI fragments."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from .data import load_curated_samples, sample_label


LABELS_VI = {
    "Mode": "Chế độ",
    "Curated Investigation": "Khám phá mẫu chuẩn (Curated)",
    "Custom Upload (coming soon)": "Tải video riêng (Custom Upload)",
    "Guided Presentation Mode": "Chế độ thuyết trình có hướng dẫn",
    "Detail level": "Mức chi tiết",
    "Simple": "Dễ hiểu",
    "Technical": "Kỹ thuật",
    "Selected curated case": "Mẫu chuẩn đang chọn",
    "Ground truth": "Nhãn đúng (Ground truth)",
    "Truth": "Nhãn đúng",
    "Raw frames": "Số frame gốc",
    "Normalized shape": "Shape sau chuẩn hoá",
    "Pose score": "Điểm pose",
    "Pose reliability": "Độ tin cậy pose",
    "RF prediction": "Dự đoán RF",
    "DL prediction": "Dự đoán DL",
    "RF confidence": "Độ tự tin RF",
    "Technique quality": "Chất lượng kỹ thuật",
    "DTW similarity": "Độ tương đồng DTW",
    "DL confidence": "Độ tự tin DL",
    "True label": "Nhãn đúng",
    "Predicted label": "Nhãn dự đoán",
    "Top probability": "Xác suất cao nhất",
    "Predicted class": "Class dự đoán",
    "Original RF": "RF ban đầu",
    "Missing joints": "Khớp bị mất",
    "Max jump": "Bước nhảy lớn nhất",
    "Decision Tree accuracy": "Accuracy của Decision Tree",
    "Tree depth": "Độ sâu cây",
    "Leaves": "Số lá",
    "Curated samples": "Số mẫu chuẩn",
    "Stroke classes": "Số lớp cú đánh",
    "Skeleton rows": "Số dòng skeleton",
    "CSV files": "Số file CSV",
    "Baseline RF": "RF gốc",
    "Degraded RF": "RF sau nhiễu",
    "Cached DL": "DL cached",
    "Degraded confidence": "Confidence sau nhiễu",
    "Label changed": "Đổi nhãn?",
    "Label changed?": "Đổi nhãn?",
    "Confidence delta": "Độ đổi confidence",
    "Quality subset samples": "Số mẫu quality subset",
    "Quality MAE": "Quality MAE",
    "Spearman rs": "Spearman rs",
    "Manual mean": "Điểm manual trung bình",
}

TECH_TERMS = {
    "RF": "Random Forest — mô hình cây quyết định tổ hợp, mạnh với feature thủ công.",
    "DL": "Deep Learning — nhánh GCN + BiLSTM + Attention học trực tiếp từ chuỗi skeleton.",
    "GCN": "Graph Convolutional Network — học quan hệ không gian giữa các khớp.",
    "BiLSTM": "Bidirectional LSTM — đọc chuyển động theo cả chiều thời gian trước và sau.",
    "Attention": "Cơ chế gán trọng số cho frame quan trọng trong cú đánh.",
    "COCO-17": "Bộ 17 keypoint cơ thể dùng làm chuẩn skeleton 2D.",
    "MediaPipe Pose": "Thư viện trích xuất keypoint cơ thể từ video từng frame.",
    "Pose QC": "Pose Quality Control — kiểm tra mất khớp, nhảy khớp, độ tin cậy skeleton.",
    "DTW": "Dynamic Time Warping — so khớp hai chuỗi chuyển động dù tốc độ thực hiện khác nhau.",
    "F1 macro": "Trung bình F1 đều cho mọi lớp; nhạy với lớp yếu.",
    "Accuracy": "Tỉ lệ dự đoán đúng trên toàn bộ mẫu.",
}


def vi_label(label: str) -> str:
    """Return Vietnamese-friendly copy while preserving the technical term."""
    return LABELS_VI.get(label, label)


def inject_design_system() -> None:
    """Apply a cyber-observatory visual layer without touching Streamlit internals too much."""
    st.markdown(
        """
        <style>
        :root {
            --bsqa-cyan: #0891b2;
            --bsqa-cyan-soft: rgba(8, 145, 178, .13);
            --bsqa-violet: #7c3aed;
            --bsqa-emerald: #059669;
            --bsqa-bg: #f6f9ff;
            --bsqa-text: #0f172a;
            --bsqa-muted: #64748b;
            --bsqa-card: rgba(255, 255, 255, .84);
            --bsqa-card-strong: rgba(255, 255, 255, .95);
            --bsqa-border: rgba(15, 23, 42, .10);
            --bsqa-shadow: 0 18px 48px rgba(15, 23, 42, .10);
        }
        .stApp {
            background:
                radial-gradient(circle at 22% 0%, rgba(34, 211, 238, .18), transparent 32rem),
                radial-gradient(circle at 88% 10%, rgba(124, 58, 237, .12), transparent 28rem),
                linear-gradient(180deg, #fbfdff 0%, var(--bsqa-bg) 48%, #eef6ff 100%);
            color: var(--bsqa-text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        }
        .block-container { max-width: 1180px; padding-top: 1.15rem; padding-bottom: 3rem; }
        .stMarkdown, .stText, p, li, label, [data-testid="stMarkdownContainer"] { font-family: inherit; }
        h1, h2, h3 { letter-spacing: -0.035em; color: var(--bsqa-text); }
        h1 { font-weight: 850; line-height: 1.02; }
        h2, h3 { font-weight: 760; }
        [data-testid="stCaptionContainer"], .stCaptionContainer { color: var(--bsqa-muted) !important; }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 50% 0%, rgba(34, 211, 238, .16), transparent 18rem),
                linear-gradient(180deg, #07111f 0%, #0f172a 54%, #111827 100%);
            border-right: 1px solid rgba(103, 232, 249, .16);
        }
        [data-testid="stSidebar"] * { color: #e5f6ff; }
        [data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, .16); }
        [data-testid="stSidebar"] a {
            border-radius: 11px;
            margin: 2px 8px;
            transition: background .18s ease, transform .18s ease;
        }
        [data-testid="stSidebar"] a:hover { background: rgba(34, 211, 238, .10); transform: translateX(2px); }
        [data-testid="stSidebar"] a[aria-current="page"] { background: rgba(148, 163, 184, .18); }
        [data-testid="stSidebar"] a[href$="/"] p,
        [data-testid="stSidebar"] a[href$="Full_Pipeline_Demo"] p,
        [data-testid="stSidebar"] a[href$="Pose_Inspector"] p,
        [data-testid="stSidebar"] a[href$="Deep_Learning_Inspector"] p,
        [data-testid="stSidebar"] a[href$="Data_Mining_Motion_Lab"] p,
        [data-testid="stSidebar"] a[href$="Error_Analysis_Lab"] p,
        [data-testid="stSidebar"] a[href$="Training_and_Evaluation"] p,
        [data-testid="stSidebar"] a[href$="Dataset_Explorer"] p,
        [data-testid="stSidebar"] a[href$="Robustness_Experiment"] p,
        [data-testid="stSidebar"] a[href$="Custom_Upload"] p { font-size: 0 !important; }
        [data-testid="stSidebar"] a[href$="/"] p::after { content: "Trang chủ"; font-size: .94rem; font-weight: 650; }
        [data-testid="stSidebar"] a[href$="Full_Pipeline_Demo"] p::after { content: "Demo pipeline đầy đủ"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Pose_Inspector"] p::after { content: "Kiểm tra skeleton"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Deep_Learning_Inspector"] p::after { content: "Giải thích Deep Learning"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Data_Mining_Motion_Lab"] p::after { content: "Data Mining Motion Lab"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Error_Analysis_Lab"] p::after { content: "Phân tích lỗi"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Training_and_Evaluation"] p::after { content: "Huấn luyện & đánh giá"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Dataset_Explorer"] p::after { content: "Khám phá dữ liệu"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Robustness_Experiment"] p::after { content: "Thử độ bền mô hình"; font-size: .94rem; }
        [data-testid="stSidebar"] a[href$="Custom_Upload"] p::after { content: "Tải video riêng"; font-size: .94rem; }

        div[data-testid="stMetric"], .bsqa-metric-card {
            border: 1px solid var(--bsqa-border);
            border-radius: 18px;
            padding: .9rem 1rem;
            background: linear-gradient(135deg, var(--bsqa-card-strong), rgba(236, 254, 255, .72));
            box-shadow: var(--bsqa-shadow);
        }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--bsqa-muted) !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--bsqa-text) !important; font-weight: 800; }
        .bsqa-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .85rem; margin: .25rem 0 .75rem; }
        .bsqa-metric-card { min-height: 104px; display: flex; flex-direction: column; justify-content: center; gap: .35rem; overflow-wrap: anywhere; }
        .bsqa-metric-label { color: var(--bsqa-muted); font-size: .78rem; line-height: 1.2; font-weight: 700; letter-spacing: .02em; }
        .bsqa-metric-value { color: var(--bsqa-text); font-size: clamp(1.05rem, 2.2vw, 1.55rem); line-height: 1.12; font-weight: 850; letter-spacing: -.035em; }
        .bsqa-metric-help { color: var(--bsqa-muted); font-size: .72rem; line-height: 1.25; }
        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(14, 165, 233, .22);
            box-shadow: 0 12px 28px rgba(15, 23, 42, .06);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stImage"], [data-testid="stVideo"] {
            border-radius: 18px;
            overflow: hidden;
        }
        .bsqa-glossary {
            margin-top: 2rem;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(34, 211, 238, 0.28);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.90));
            color: #e2e8f0;
            box-shadow: 0 20px 55px rgba(2, 6, 23, .16);
        }
        .bsqa-glossary h4 { margin: 0 0 .7rem 0; color: #a5f3fc; }
        .bsqa-glossary ol { margin-bottom: 0; }
        .bsqa-glossary li { margin: .38rem 0; }
        .bsqa-footnote {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.3rem;
            height: 1.3rem;
            padding: 0 .35rem;
            border-radius: 999px;
            margin-left: .25rem;
            background: rgba(34, 211, 238, .15);
            color: #0e7490;
            border: 1px solid rgba(14, 116, 144, .25);
            font-size: .72rem;
            font-weight: 800;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bsqa-cyan: #22d3ee;
                --bsqa-cyan-soft: rgba(34, 211, 238, .16);
                --bsqa-violet: #a78bfa;
                --bsqa-emerald: #34d399;
                --bsqa-bg: #020617;
                --bsqa-text: #e5edf8;
                --bsqa-muted: #94a3b8;
                --bsqa-card: rgba(15, 23, 42, .74);
                --bsqa-card-strong: rgba(15, 23, 42, .92);
                --bsqa-border: rgba(148, 163, 184, .20);
                --bsqa-shadow: 0 18px 50px rgba(0, 0, 0, .35);
            }
            .stApp {
                background:
                    radial-gradient(circle at 15% 0%, rgba(34, 211, 238, .16), transparent 28rem),
                    radial-gradient(circle at 88% 16%, rgba(167, 139, 250, .16), transparent 24rem),
                    linear-gradient(180deg, #020617 0%, #0b1220 56%, #020617 100%);
            }
            [data-testid="stAlert"] { background: rgba(15, 23, 42, .72); color: var(--bsqa-text); }
            div[data-testid="stMetric"], .bsqa-metric-card {
                background: linear-gradient(135deg, rgba(15, 23, 42, .92), rgba(8, 47, 73, .48));
            }
            .bsqa-footnote { color: #67e8f9; border-color: rgba(103, 232, 249, .35); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Optional[str]:
    inject_design_system()
    st.sidebar.title("BSQAv2 Observatory")
    mode = st.sidebar.radio(
        vi_label("Mode"),
        ["Curated Investigation", "Custom Upload (coming soon)"],
        index=0,
        format_func=vi_label,
        help="Curated = mẫu đã kiểm tra sẵn; Custom Upload = chạy pipeline trên video của bạn.",
    )
    st.sidebar.toggle(vi_label("Guided Presentation Mode"), value=True, key="guided_mode")
    st.sidebar.selectbox(vi_label("Detail level"), ["Simple", "Technical"], format_func=vi_label, key="detail_level")

    if mode != "Curated Investigation":
        st.sidebar.info("Custom Upload đã có trang beta riêng. Dùng các trang Curated cho demo ổn định nhất.")
        render_sidebar_glossary()
        return None

    samples = load_curated_samples()
    if not samples:
        st.sidebar.error("Chưa có mẫu curated. Hãy build webapp/artifacts/curated/manifest.json trước.")
        render_sidebar_glossary()
        return None

    sample_by_label = {sample_label(sample): sample.sample_id for sample in samples}
    labels = list(sample_by_label.keys())
    selected_label = st.sidebar.selectbox(vi_label("Selected curated case"), labels)
    render_sidebar_glossary()
    return sample_by_label[selected_label]


def render_sidebar_glossary() -> None:
    with st.sidebar.expander("Thuật ngữ nhanh", expanded=False):
        st.caption("Giữ nguyên technical term, giải thích ngắn bằng tiếng Việt.")
        for term in ["RF", "DL", "GCN", "BiLSTM", "Attention", "Pose QC", "DTW"]:
            st.markdown(f"**{term}** — {TECH_TERMS[term]}")


def explanation_card(title: str, body: str) -> None:
    if st.session_state.get("guided_mode", True):
        st.info(f"**{title}**\n\n{body}")


def metric_row(metrics: List[tuple]) -> None:
    cards = []
    for label, value, help_text in metrics:
        cards.append(
            f'<div class="bsqa-metric-card" title="{escape(str(help_text))}">'
            f'<div class="bsqa-metric-label">{escape(vi_label(label))}</div>'
            f'<div class="bsqa-metric-value">{escape(str(value))}</div>'
            f'<div class="bsqa-metric-help">{escape(str(help_text))}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="bsqa-metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def show_video(video_path: str) -> None:
    path = Path(video_path)
    if path.exists():
        st.video(str(path))
    else:
        st.warning(f"Không tìm thấy video: {video_path}")


def render_three_hero() -> None:
    """Render a lightweight Three.js skeleton/neural-field hero for the home page."""
    components.html(
        """
        <style>
          html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; }
          .bsqa-hero-wrap {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            border: 1px solid rgba(34, 211, 238, .34);
            background:
              radial-gradient(circle at 18% 12%, rgba(34, 211, 238, .26), transparent 32%),
              radial-gradient(circle at 84% 18%, rgba(139, 92, 246, .24), transparent 30%),
              linear-gradient(135deg, #020617 0%, #0f172a 56%, #111827 100%);
            height: 340px;
            box-shadow: 0 24px 80px rgba(2, 6, 23, .28), inset 0 1px 0 rgba(255,255,255,.08);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
          }
          .bsqa-hero-wrap::before {
            content: "";
            position: absolute; inset: 0;
            background-image: linear-gradient(rgba(103,232,249,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(103,232,249,.07) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: radial-gradient(circle at 70% 52%, black, transparent 72%);
          }
          .bsqa-hero-copy {
            position: absolute; z-index: 2; left: 32px; top: 28px; max-width: 565px; color: #f8fafc;
          }
          .bsqa-kicker { color: #67e8f9; font-weight: 800; letter-spacing: .10em; text-transform: uppercase; font-size: .78rem; }
          .bsqa-hero-copy h1 { margin: .45rem 0 .65rem; font-size: clamp(2.15rem, 5.2vw, 4.15rem); line-height: .94; letter-spacing: -.055em; font-weight: 900; }
          .bsqa-hero-copy p { margin: 0; color: #cbd5e1; font-size: 1.02rem; line-height: 1.58; max-width: 52ch; }
          .bsqa-pill-row { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.05rem; }
          .bsqa-pill { border: 1px solid rgba(103,232,249,.32); border-radius: 999px; padding: .42rem .74rem; background: rgba(15,23,42,.56); color: #e0f2fe; backdrop-filter: blur(10px); font-size: .85rem; box-shadow: inset 0 1px 0 rgba(255,255,255,.06); }
          #bsqa-three-canvas { position: absolute; inset: 0; z-index: 1; }
          @media (prefers-color-scheme: light) {
            .bsqa-hero-wrap { box-shadow: 0 24px 80px rgba(15, 23, 42, .18), inset 0 1px 0 rgba(255,255,255,.08); }
          }
          @media (max-width: 720px) {
            .bsqa-hero-wrap { height: 440px; }
            .bsqa-hero-copy { left: 18px; right: 18px; top: 18px; }
          }
        </style>
        <div class="bsqa-hero-wrap">
          <div class="bsqa-hero-copy">
            <div class="bsqa-kicker">Badminton Stroke Quality Assessment</div>
            <h1>Phòng quan sát chuyển động cầu lông</h1>
            <p>
              Từ video thật → MediaPipe Pose → COCO-17 Skeleton → Random Forest + GCN/BiLSTM/Attention.
              Giao diện này ưu tiên tiếng Việt, nhưng vẫn giữ technical term để thuyết trình đúng chuyên môn.
            </p>
            <div class="bsqa-pill-row">
              <span class="bsqa-pill">3D neural skeleton</span>
              <span class="bsqa-pill">Pose QC</span>
              <span class="bsqa-pill">RF evidence</span>
              <span class="bsqa-pill">DL Attention</span>
              <span class="bsqa-pill">DTW quality</span>
            </div>
          </div>
          <canvas id="bsqa-three-canvas" style="width:100%;height:340px;display:block;"></canvas>
        </div>
        <script src="https://unpkg.com/three@0.159.0/build/three.min.js"></script>
        <script>
        (function(){
          const canvas = document.getElementById('bsqa-three-canvas');
          if (!canvas || !window.THREE) return;
          const scene = new THREE.Scene();
          const renderer = new THREE.WebGLRenderer({canvas, alpha:true, antialias:true});
          const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
          camera.position.set(0, 0.4, 7);
          const group = new THREE.Group();
          scene.add(group);

          const points = [
            [0,1.7,0],[-.35,1.35,0],[.35,1.35,0],[-.8,.8,0],[.8,.8,0],[-1.1,.15,0],[1.1,.15,0],
            [-.25,.45,0],[.25,.45,0],[-.45,-.45,0],[.45,-.45,0],[-.55,-1.35,0],[.55,-1.35,0]
          ].map(p=>new THREE.Vector3(...p));
          const edges = [[0,1],[0,2],[1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]];
          const nodeMat = new THREE.MeshBasicMaterial({color:0x67e8f9});
          const hotMat = new THREE.MeshBasicMaterial({color:0xa78bfa});
          const lineMat = new THREE.LineBasicMaterial({color:0x22d3ee, transparent:true, opacity:.72});
          points.forEach((p, i)=>{
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(i === 6 ? .105 : .075, 24, 24), i === 6 ? hotMat : nodeMat);
            mesh.position.copy(p); group.add(mesh);
          });
          edges.forEach(([a,b])=>{
            const geo = new THREE.BufferGeometry().setFromPoints([points[a], points[b]]);
            group.add(new THREE.Line(geo, lineMat));
          });
          const particleGeo = new THREE.BufferGeometry();
          const verts = [];
          for (let i=0;i<360;i++) {
            const r = 2.2 + Math.random()*1.8, t = Math.random()*Math.PI*2, y = (Math.random()-.5)*3.2;
            verts.push(Math.cos(t)*r, y, Math.sin(t)*r);
          }
          particleGeo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
          const particles = new THREE.Points(particleGeo, new THREE.PointsMaterial({color:0x8b5cf6,size:.025,transparent:true,opacity:.75}));
          scene.add(particles);
          function resize(){
            const rect = canvas.getBoundingClientRect();
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.setSize(rect.width, rect.height, false);
            camera.aspect = rect.width / rect.height; camera.updateProjectionMatrix();
          }
          resize(); window.addEventListener('resize', resize);
          const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          function tick(ms){
            const t = ms * 0.001;
            group.rotation.y = reduce ? -0.35 : Math.sin(t*.45)*0.35 - 0.35;
            group.rotation.x = reduce ? 0.08 : Math.sin(t*.7)*0.08;
            group.position.x = 1.6;
            particles.rotation.y = reduce ? 0 : t*.045;
            renderer.render(scene, camera);
            if (!reduce) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        })();
        </script>
        """,
        height=360,
    )


def glossary_badge(number: int) -> str:
    return f'<span class="bsqa-footnote">{number}</span>'


def render_glossary_footer(terms: Optional[Iterable[str]] = None) -> None:
    selected_terms = list(terms or ["RF", "DL", "GCN", "BiLSTM", "Attention", "COCO-17", "MediaPipe Pose", "Pose QC", "DTW", "F1 macro"])
    items = "".join(
        f"<li><strong>{escape(term)}</strong>: {escape(TECH_TERMS[term])}</li>"
        for term in selected_terms
        if term in TECH_TERMS
    )
    st.markdown(
        f"""
        <div class="bsqa-glossary">
          <h4>Bảng chú giải thuật ngữ</h4>
          <ol>{items}</ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
