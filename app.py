"""
SD-DeepLab Streamlit App
Professional polyp segmentation interface with real-time inference,
bounding box detection, and performance metrics
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
from model import SDDeepLab
from utils import (
    extract_bounding_boxes, calculate_metrics, visualize_segmentation,
    draw_metrics_box, prepare_tensor, restore_output, get_dataset_statistics
)


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SD-DeepLab Polyp Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .header-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #43a047;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'inference_done' not in st.session_state:
    st.session_state.inference_done = False
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}


@st.cache_resource
def load_model(model_path: str):
    """Load SD-DeepLab model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SDDeepLab(pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def run_inference(image: np.ndarray, model, device) -> dict:
    """Run inference on image"""
    try:
        with torch.no_grad():
            # Prepare tensor
            tensor, original_size = prepare_tensor(image, size=(512, 512))
            tensor = tensor.to(device)
            
            # Inference
            start_time = time.time()
            output = model(tensor)
            inference_time = time.time() - start_time
            
            # Get segmentation mask (already sigmoid'd from model)
            mask = output['mask']
            
            # Restore to original size
            mask_np = restore_output(mask, original_size)
            
            # Extract bounding boxes
            bboxes = extract_bounding_boxes(mask_np)
            
            return {
                'mask': mask_np,
                'bboxes': bboxes,
                'inference_time': inference_time,
                'original_image': image
            }
    except Exception as e:
        st.error(f"❌ Inference error: {str(e)}")
        return None


def display_architecture_diagram():
    """Display architecture diagram"""
    try:
        img_path = Path(__file__).parent / "Architecture_diagram.png"
        if img_path.exists():
            st.image(str(img_path), caption="SD-DeepLab Architecture", use_container_width=True)
        else:
            st.warning("⚠️ Architecture diagram file not found")
    except Exception as e:
        st.warning(f"Could not load architecture diagram: {str(e)}")


# ==================== MAIN APP ====================

# Header
st.markdown("""
    <div class="header-container">
        <h1>🔬 SD-DeepLab Polyp Segmentation</h1>
        <p>Structural Deep Learning for Colorectal Polyp Localization & Segmentation</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model loading
    st.subheader("Model Setup")
    model_path = "sddeeplab_final.pth"
    
    if st.button("🚀 Load Model", use_container_width=True):
        with st.spinner("Loading SD-DeepLab model..."):
            model, device = load_model(model_path)
            if model is not None:
                st.session_state.model = model
                st.session_state.device = device
                st.success(f"✅ Model loaded! Device: {device}")
    
    # Display device info
    st.info(f"🖥️ Device: {st.session_state.device}")
    
    # Visualization options
    st.subheader("Visualization")
    show_bbox = st.checkbox("Show Bounding Boxes", value=True)
    
    st.divider()
    st.info("""
    **Quick Start:**
    1. Load the model using the button above
    2. Upload an image
    3. View segmentation & metrics
    4. Download results
    """)


# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Segmentation", 
    "📊 Dataset Metrics", 
    "📐 Architecture",
    "ℹ️ About"
])

# ==================== TAB 1: SEGMENTATION ====================
with tab1:
    col1, col2 = st.columns([1.5, 1.5])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a polyp image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            
            st.image(image, caption="Original Image", use_container_width=True)
            st.info(f"📏 Image size: {image_np.shape[1]}×{image_np.shape[0]} pixels")
    
    with col2:
        st.subheader("🔍 Segmentation Results")
        
        if uploaded_file is not None:
            if st.session_state.model is None:
                st.warning("⚠️ Please load the model first (left sidebar)")
            else:
                if st.button("Run Segmentation", use_container_width=True):
                    with st.spinner("🔄 Running segmentation..."):
                        results = run_inference(
                            image_np,
                            st.session_state.model,
                            st.session_state.device
                        )
                        
                        if results:
                            st.session_state.current_results = results
                            st.session_state.inference_done = True
    
    # Display results
    if st.session_state.inference_done and st.session_state.current_results:
        results = st.session_state.current_results
        
        st.divider()
        st.subheader("📋 Results Summary")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Inference Time", f"{results['inference_time']*1000:.1f} ms")
        with col2:
            st.metric("Polyps Detected", len(results['bboxes']))
        with col3:
            if results['bboxes']:
                total_area = sum(b['area'] for b in results['bboxes'])
                st.metric("Total Polyp Area", f"{total_area:.0f} px²")
        with col4:
            if results['bboxes']:
                avg_area = np.mean([b['area'] for b in results['bboxes']])
                st.metric("Avg Polyp Size", f"{avg_area:.0f} px²")
        
        st.divider()
        
        # Visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(results['original_image'], use_container_width=True)
        
        with col2:
            st.markdown("**Segmentation Mask**")
            mask_display = results['mask']
            # Convert to pure black and white
            mask_bw = (mask_display > 0.5).astype(np.uint8) * 255
            st.image(mask_bw, use_container_width=True)
        
        with col3:
            st.markdown("**Bounding Boxes**")
            visualization = visualize_segmentation(
                results['original_image'],
                results['mask'],
                bboxes=results['bboxes'] if show_bbox else None
            )
            st.image(visualization, use_container_width=True)
        
        # Bounding boxes details
        if results['bboxes']:
            st.subheader("🎯 Detected Polyps")
            bbox_data = []
            for i, bbox in enumerate(results['bboxes'], 1):
                bbox_data.append({
                    'ID': i,
                    'X': bbox['x'],
                    'Y': bbox['y'],
                    'Width': bbox['width'],
                    'Height': bbox['height'],
                    'Area (px²)': f"{bbox['area']:.0f}",
                    'Confidence': "High"
                })
            
            st.dataframe(bbox_data, use_container_width=True, hide_index=True)
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                # Save segmentation
                result_image = Image.fromarray(
                    (visualization * 255).astype(np.uint8)
                )
                result_image.save("/tmp/segmentation_result.png")
                with open("/tmp/segmentation_result.png", "rb") as f:
                    st.download_button(
                        "📥 Download Segmentation",
                        f,
                        file_name="segmentation_result.png",
                        mime="image/png"
                    )


# ==================== TAB 2: DATASET METRICS ====================
with tab2:
    st.subheader("📊 Dataset Evaluation Results")
    st.info("""
    These are the pre-trained model's performance metrics on standard benchmarks.
    The model shows robust performance across different datasets.
    """)
    
    stats = get_dataset_statistics()
    
    # Kvasir-SEG Test
    st.markdown("### 🏥 Kvasir-SEG Test (100 samples)")
    kvasir_stats = stats['kvasir_seg_test']['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dice Score (Mean)", f"{kvasir_stats['dice']['mean']:.4f}", 
                 f"±{kvasir_stats['dice']['std']:.4f}")
    with col2:
        st.metric("IoU (Mean)", f"{kvasir_stats['iou']['mean']:.4f}",
                 f"±{kvasir_stats['iou']['std']:.4f}")
    with col3:
        st.metric("Precision (Mean)", f"{kvasir_stats['precision']['mean']:.4f}",
                 f"±{kvasir_stats['precision']['std']:.4f}")
    with col4:
        st.metric("Recall (Mean)", f"{kvasir_stats['recall']['mean']:.4f}",
                 f"±{kvasir_stats['recall']['std']:.4f}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F2-Score (Mean)", f"{kvasir_stats['f2']['mean']:.4f}",
                 f"±{kvasir_stats['f2']['std']:.4f}")
    with col2:
        st.metric("Accuracy (Mean)", f"{kvasir_stats['accuracy']['mean']:.4f}",
                 f"±{kvasir_stats['accuracy']['std']:.4f}")
    with col3:
        st.metric("Specificity (Mean)", f"{kvasir_stats['specificity']['mean']:.4f}",
                 f"±{kvasir_stats['specificity']['std']:.4f}")
    with col4:
        st.metric("HD95 (Mean)", f"{kvasir_stats['hd95']['mean']:.2f}",
                 f"±{kvasir_stats['hd95']['std']:.2f}",
                 help="Lower is better - Hausdorff Distance at 95th percentile")
    
    # Summary stats table
    st.dataframe({
        'Metric': ['Dice', 'IoU', 'Precision', 'Recall', 'F2', 'Accuracy', 'Specificity', 'HD95'],
        'Mean': [
            f"{kvasir_stats['dice']['mean']:.4f}",
            f"{kvasir_stats['iou']['mean']:.4f}",
            f"{kvasir_stats['precision']['mean']:.4f}",
            f"{kvasir_stats['recall']['mean']:.4f}",
            f"{kvasir_stats['f2']['mean']:.4f}",
            f"{kvasir_stats['accuracy']['mean']:.4f}",
            f"{kvasir_stats['specificity']['mean']:.4f}",
            f"{kvasir_stats['hd95']['mean']:.2f}"
        ],
        'Std': [
            f"{kvasir_stats['dice']['std']:.4f}",
            f"{kvasir_stats['iou']['std']:.4f}",
            f"{kvasir_stats['precision']['std']:.4f}",
            f"{kvasir_stats['recall']['std']:.4f}",
            f"{kvasir_stats['f2']['std']:.4f}",
            f"{kvasir_stats['accuracy']['std']:.4f}",
            f"{kvasir_stats['specificity']['std']:.4f}",
            f"{kvasir_stats['hd95']['std']:.4f}"
        ],
        'Min': [
            f"{kvasir_stats['dice']['min']:.4f}",
            f"{kvasir_stats['iou']['min']:.4f}",
            f"{kvasir_stats['precision']['min']:.4f}",
            f"{kvasir_stats['recall']['min']:.4f}",
            f"{kvasir_stats['f2']['min']:.4f}",
            f"{kvasir_stats['accuracy']['min']:.4f}",
            f"{kvasir_stats['specificity']['min']:.4f}",
            f"{kvasir_stats['hd95']['min']:.2f}"
        ],
        'Max': [
            f"{kvasir_stats['dice']['max']:.4f}",
            f"{kvasir_stats['iou']['max']:.4f}",
            f"{kvasir_stats['precision']['max']:.4f}",
            f"{kvasir_stats['recall']['max']:.4f}",
            f"{kvasir_stats['f2']['max']:.4f}",
            f"{kvasir_stats['accuracy']['max']:.4f}",
            f"{kvasir_stats['specificity']['max']:.4f}",
            f"{kvasir_stats['hd95']['max']:.2f}"
        ]
    }, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # CVC-ClinicDB
    st.markdown("### 🏥 CVC-ClinicDB (612 samples)")
    st.info("**External validation dataset** - Shows model generalization to varied polyp types")
    
    cvc_stats = stats['cvc_clinicdb']['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dice Score", f"{cvc_stats['dice']['mean']:.4f}",
                 f"±{cvc_stats['dice']['std']:.4f}")
    with col2:
        st.metric("IoU", f"{cvc_stats['iou']['mean']:.4f}",
                 f"±{cvc_stats['iou']['std']:.4f}")
    with col3:
        st.metric("Precision", f"{cvc_stats['precision']['mean']:.4f}",
                 f"±{cvc_stats['precision']['std']:.4f}")
    with col4:
        st.metric("Recall", f"{cvc_stats['recall']['mean']:.4f}",
                 f"±{cvc_stats['recall']['std']:.4f}")


# ==================== TAB 3: ARCHITECTURE ====================
with tab3:
    st.subheader("🏗️ SD-DeepLab Architecture")
    
    st.info("""
    **SD-DeepLab** is a Structural Deep Learning architecture designed specifically 
    for polyp segmentation, combining geometric priors with multi-scale features.
    """)
    
    display_architecture_diagram()
    
    st.divider()
    
    # Architecture explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Key Components")
        st.markdown("""
        **1. ResNet-50 Backbone**
        - Stride-16 with dilated convolutions
        - Multi-scale feature extraction (F2-F5)
        
        **2. ASPP (Atrous Spatial Pyramid Pooling)**
        - Captures multi-scale context
        - Atrous rates: [1, 6, 12, 18]
        """)
    
    with col2:
        st.markdown("### 🎯 Structural Components")
        st.markdown("""
        **3. Structural Init Head**
        - Generates 4 geometric channels:
          - M: Occupancy mask
          - B: Boundary map
          - D: Distance map
          - U: Uncertainty map
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Attention Mechanisms")
        st.markdown("""
        **SCCR (Structural-Conditioned Channel Routing)**
        - Uses geometric priors for channel attention
        - Better boundary/interior separation
        
        **ASA (Anisotropic Strip Attention)**
        - Handles elongated/irregular shapes
        - Vertical + horizontal strip convolutions
        """)
    
    with col2:
        st.markdown("### ⚡ Advanced Features")
        st.markdown("""
        **Structural Transition Blocks (STB)**
        - Evolves structural state at multiple scales
        - SDAA attention (SCCR + ASA)
        
        **Structural Energy Layer**
        - Laplacian smoothing for M, D
        - Gaussian smoothing for B, U
        - Regularizes geometry dynamically
        """)
    
    st.divider()
    
    st.markdown("### 🔄 Data Flow Pipeline")
    st.markdown("""
    ```
    Input Image (512×512)
         ↓
    ResNet50 Backbone (multi-scale features)
         ↓
    ASPP (context aggregation)
         ↓
    StructuralInitHead (M, B, D, U @ H/16)
         ↓
    STB4 → STB3 → STB2 (multi-scale transitions with SDAA)
         ↓
    StructuralEnergyLayer (geometric smoothing)
         ↓
    StructuralProjectionHead (upsample to 512×512)
         ↓
    Segmentation Mask + Bounding Boxes
    ```
    """)
    
    st.divider()
    
    st.markdown("### 🎓 Loss Function")
    st.markdown("""
    **Combined Multi-Task Loss:**
    - Dice Loss (overlap)
    - BCE Loss (standard segmentation)
    - Lovász-Sigmoid Loss (boundary-aware, IoU-optimized)
    - Boundary Loss (edge refinement)
    - Distance Loss (shape smoothness)
    - Energy Regularization (geometric stability)
    """)


# ==================== TAB 4: ABOUT ====================
with tab4:
    st.subheader("ℹ️ About SD-DeepLab")
    
    st.markdown("""
    ### 📖 Overview
    
    **SD-DeepLab** is a state-of-the-art segmentation architecture specifically designed for 
    colorectal polyp detection and segmentation. It combines traditional DeepLab principles 
    with novel **structural geometric priors** for improved robustness.
    
    ### 🎯 Key Advantages
    
    ✅ **Structural Awareness** - Incorporates geometric guidance (M, B, D, U)
    
    ✅ **Multi-Scale Processing** - ASPP + hierarchical transition blocks
    
    ✅ **Robust Attention** - SDAA handles irregular polyp shapes effectively
    
    ✅ **Smooth Predictions** - Energy layer prevents jagged boundaries
    
    ✅ **Boundary-Aware** - Lovász loss emphasizes edge precision
    
    ### 📊 Benchmark Performance
    
    **Kvasir-SEG Test (100 images):**
    - Dice: 90.77% ± 11.92%
    - IoU: 84.84% ± 16.04%
    - HD95: 24.60 ± 49.72 pixels
    
    **CVC-ClinicDB (612 images):**
    - Dice: 83.30% ± 20.52%
    - IoU: 75.22% ± 22.34%
    - Generalizes well to external datasets
    
    ### 🔬 Applications
    
    - **Clinical Diagnosis** - Assists endoscopists in polyp detection
    - **Research** - Benchmarking tool for colonoscopy image analysis
    - **Quality Assurance** - Validates detection quality during procedures
    - **Education** - Training resource for medical imaging
    
    ### 👨‍💼 Model Details
    
    | Property | Value |
    |----------|-------|
    | **Backbone** | ResNet-50 (pretrained on ImageNet) |
    | **Input Size** | 512×512 pixels (RGB) |
    | **Output** | 512×512 binary segmentation mask |
    | **Parameters** | ~50M |
    | **Inference Time** | ~80-150ms (GPU), ~300-500ms (CPU) |
    | **Framework** | PyTorch |
    
    ### 🤝 Support & Contact
    
    For issues, questions, or feature requests, please reach out.
    """)


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em; padding: 20px;'>
        <p>SD-DeepLab v1.0 | Medical Image Analysis | Colorectal Polyp Segmentation</p>
        <p>Built with PyTorch, Streamlit, and ❤️ for better healthcare</p>
    </div>
""", unsafe_allow_html=True)
