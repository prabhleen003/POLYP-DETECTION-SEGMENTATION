# 🔬 SD-DeepLab Polyp Segmentation Streamlit App

A professional, production-ready Streamlit application for colorectal polyp detection and segmentation using the advanced **SD-DeepLab** (Structural DeepLab) deep learning architecture.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Segmentation**: Upload polyp images and get instant segmentation masks
- **Bounding Box Detection**: Automatically extract bounding boxes from segmentation masks
- **Multi-Image Upload**: Process multiple images sequentially
- **High Performance**: Optimized inference on both GPU and CPU

### 📊 Analytics & Metrics
- **Pre-computed Dataset Metrics**: 
  - Kvasir-SEG Test (100 samples): Dice 90.77%, IoU 84.84%
  - CVC-ClinicDB (612 samples): Dice 83.30%, IoU 75.22%
- **Per-Image Metrics**: Dice, IoU, Precision, Recall, F2-Score, Accuracy, Specificity, HD95
- **Detailed Statistics**: Mean, Std, Min, Max across datasets

### 📐 Architecture Visualization
- **Architecture Diagram**: Visual representation of SD-DeepLab pipeline
- **Component Explanations**: Detailed descriptions of each module:
  - ResNet-50 Backbone
  - ASPP (Atrous Spatial Pyramid Pooling)
  - SCCR (Structural-Conditioned Channel Routing)
  - ASA (Anisotropic Strip Attention)
  - SDAA (Structural Dual-Axis Attention)
  - Structural Transition Blocks (STB)
  - Structural Energy Layer
  - Projection Head

### 🎨 Visualization Options
- **Overlay Transparency Control**: Adjust segmentation overlay visibility
- **Bounding Box Visualization**: Display detected polyps with coordinates
- **Side-by-side Comparison**: Original image vs. Segmentation vs. Overlay
- **Summary Statistics**: Polyp count, total area, average size

### 💾 Export & Download
- **Download Segmentation Results**: Save visualizations as PNG
- **Metrics Export**: View detailed metrics for each image
- **Bounding Box Coordinates**: Export detection data for further analysis

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 4GB RAM minimum (8GB+ recommended)

### Installation

1. **Clone/Download the Project**
   ```bash
   cd "Colorectal polyp segmentatiom"
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

The app will open in your default browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Load the Model
1. Open the app in your browser
2. Go to the left sidebar under "⚙️ Configuration"
3. Click **"🚀 Load Model"** button
4. Wait for confirmation: "✅ Model loaded!"

> **Note**: First load may take 30-60 seconds. Subsequent loads are cached.

### Step 2: Upload Image
1. Click **"📤 Upload Image"** in the Segmentation tab
2. Select a polyp image (JPG, PNG, BMP, TIFF)
3. View the original image preview

### Step 3: Run Segmentation
1. Click **"Run Segmentation"** button
2. Wait for inference (typically 80-150ms on GPU, 300-500ms on CPU)
3. View results:
   - **Original Image**: Input image
   - **Segmentation Mask**: Green mask showing detected polyp
   - **Overlay Visualization**: Mask overlaid on original with bounding boxes

### Step 4: Analyze Results
- **Metrics Summary**: Inference time, polyp count, total area, average size
- **Bounding Boxes**: Detailed table with coordinates and areas
- **Download**: Export results as PNG image

### Step 5: Compare Performance
- Go to **📊 Dataset Metrics** tab
- View pre-trained model performance on standard benchmarks
- Compare Kvasir-SEG vs CVC-ClinicDB results

### Step 6: Understand Architecture
- Go to **📐 Architecture** tab
- View detailed explanation of SD-DeepLab components
- Understand the data flow pipeline

---

## 🏗️ Project Structure

```
Colorectal polyp segmentatiom/
├── app.py                          # Main Streamlit application
├── model.py                        # SD-DeepLab architecture
├── utils.py                        # Utility functions
├── requirements.txt                # Python dependencies
├── sddeeplab_final.pth            # Pre-trained model weights
├── Architecture_diagram.png        # Architecture visualization
└── README.md                       # This file
```

### File Descriptions

#### `app.py`
Main Streamlit application with 4 tabs:
- **Segmentation Tab**: Upload images and run inference
- **Dataset Metrics Tab**: View benchmark performance
- **Architecture Tab**: Understand the model design
- **About Tab**: Project information and details

#### `model.py`
Complete SD-DeepLab architecture implementation:
- `SCCR`: Structural-Conditioned Channel Routing
- `ASA`: Anisotropic Strip Attention
- `SDAA`: Combined structural attention (SCCR + ASA)
- `StructuralInitHead`: Generates M, B, D, U channels
- `StructuralTransitionBlock`: Evolves structural state
- `StructuralEnergyLayer`: Geometric smoothing
- `ASPP`: Atrous Spatial Pyramid Pooling
- `ResNet50Backbone`: Feature extraction
- `StructuralProjectionHead`: Output prediction
- `SDDeepLab`: Complete pipeline

#### `utils.py`
Utility functions for processing and analysis:
- `binarize_mask()`: Convert soft masks to binary
- `extract_bounding_boxes()`: Get bounding boxes from masks
- `calculate_metrics()`: Compute Dice, IoU, Precision, Recall, etc.
- `calculate_hausdorff_distance()`: HD95 metric
- `visualize_segmentation()`: Create visualizations
- `prepare_tensor()`: Image preprocessing
- `restore_output()`: Resize to original dimensions
- `get_dataset_statistics()`: Return benchmark metrics

---

## 🔧 Configuration Options

### Inference Parameters (Sidebar)

**Segmentation Threshold** (0.3 - 0.9)
- Controls sensitivity of polyp detection
- Higher values = more conservative
- Default: 0.5

### Visualization Options (Sidebar)

**Show Segmentation Overlay** (Toggle)
- Display green overlay on original image
- Default: On

**Show Bounding Boxes** (Toggle)
- Display detected polyp bounding boxes
- Default: On

**Overlay Transparency** (0.1 - 0.9)
- Control opacity of segmentation overlay
- Lower = more transparent
- Default: 0.4

---

## 📊 Metrics Explained

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Dice** | Overlap between prediction and ground truth | 0-1 | Higher |
| **IoU** | Intersection over Union | 0-1 | Higher |
| **Precision** | Ratio of true positives to predicted positives | 0-1 | Higher |
| **Recall** | Ratio of true positives to actual positives | 0-1 | Higher |
| **F2-Score** | Harmonic mean (emphasizes recall) | 0-1 | Higher |
| **Accuracy** | Overall correctness | 0-1 | Higher |
| **Specificity** | True negative rate | 0-1 | Higher |
| **HD95** | Hausdorff distance at 95th percentile | 0-inf | Lower |

---

## 🏆 Model Performance

### Kvasir-SEG Test Set (100 samples)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Dice | 0.9077 | 0.1192 | 0.3924 | 0.9879 |
| IoU | 0.8484 | 0.1604 | 0.2441 | 0.9760 |
| Precision | 0.9120 | 0.1299 | 0.2443 | 0.9959 |
| Recall | 0.9310 | 0.1301 | 0.3075 | 1.0000 |
| F2-Score | 0.9175 | 0.1200 | 0.3561 | 0.9928 |
| Accuracy | 0.9704 | 0.0496 | 0.6956 | 0.9991 |
| Specificity | 0.9855 | 0.0293 | 0.7857 | 0.9997 |
| HD95 | 24.6021 | 49.7211 | 0.0000 | 269.9623 |

### CVC-ClinicDB (612 samples)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Dice | 0.8330 | 0.2052 | 0.0000 | 0.9865 |
| IoU | 0.7522 | 0.2234 | 0.0000 | 0.9733 |
| Precision | 0.8715 | 0.1515 | 0.0000 | 1.0000 |
| Recall | 0.8610 | 0.2348 | 0.0000 | 1.0000 |

---

## ⚡ Performance Tips

### GPU Acceleration
- **Recommended**: NVIDIA GPU with 2GB+ VRAM
- **Automatic Detection**: App automatically detects and uses GPU
- **Inference Speed**: ~80-150ms (vs 300-500ms on CPU)

### Optimization for Large Batches
1. Keep model loaded (don't reload unnecessarily)
2. Use threshold adjustment for different sensitivity needs
3. Process images sequentially (built-in safety)

### Memory Management
- App loads model once and caches it in memory
- Each image is processed independently
- No memory leaks on repeated inference

---

## 🐛 Troubleshooting

### Model Won't Load
- **Error**: "Error loading model"
- **Solution**: Ensure `sddeeplab_final.pth` is in the same directory
- Check file is not corrupted (verify file size > 100MB)

### Out of Memory
- **Error**: "CUDA out of memory"
- **Solution**: Either use CPU mode or reduce image size before upload
- Close other GPU-intensive applications

### Slow Inference
- **Problem**: Inference takes > 5 seconds
- **Solution**: 
  - Check if using GPU: look for device info in sidebar
  - Verify image resolution is reasonable (< 2000×2000)
  - Restart Streamlit app (sometimes helps with memory)

### Architecture Diagram Missing
- **Warning**: "Architecture diagram file not found"
- **Solution**: Ensure `Architecture_diagram.png` is in project directory

---

## 🔄 Advanced Usage

### Running with Custom Configuration

**CPU-only mode** (for servers without GPU):
```bash
CUDA_VISIBLE_DEVICES="" streamlit run app.py
```

**Specific GPU** (multi-GPU systems):
```bash
CUDA_VISIBLE_DEVICES=0 streamlit run app.py
```

**Custom port**:
```bash
streamlit run app.py --server.port 8502
```

---

## 📚 Understanding the Architecture

### Pipeline Overview

```
Input Image (512×512)
     ↓
ResNet50 Backbone (extracts multi-scale features)
     ↓
ASPP (captures multi-scale context)
     ↓
StructuralInitHead (generates M, B, D, U)
     ↓
STB4 → STB3 → STB2 (hierarchical refinement with SDAA)
     ↓
StructuralEnergyLayer (smooths geometry)
     ↓
StructuralProjectionHead (upsample to 512×512)
     ↓
Segmentation Mask + Bounding Boxes
```

### Key Innovation: Structural Priors

Unlike standard DeepLab, SD-DeepLab maintains **4 geometric channels**:

1. **M (Occupancy Mask)**: Polyp presence
2. **B (Boundary Map)**: Edge locations
3. **D (Distance Map)**: Distance from edges (-1 to 1)
4. **U (Uncertainty Map)**: Prediction confidence (0.05 to 0.95)

These priors guide attention mechanisms (SCCR, ASA) for better polyp detection.

---

## 📖 Citation

If you use this application in research, please cite:

```bibtex
@misc{sddeeplab2024,
  title={SD-DeepLab: Structural Deep Learning for Polyp Segmentation},
  year={2024},
  note={PyTorch Implementation with Streamlit Interface}
}
```

---

## 📝 License

This project is provided as-is for research and educational purposes.

---

## 🤝 Support

For questions, issues, or feature requests, please contact the development team.

### Common Questions

**Q: Can I use this on my own images?**
A: Yes! Just upload JPG, PNG, BMP, or TIFF images. The app handles preprocessing automatically.

**Q: What's the minimum image quality required?**
A: Images should be clear endoscopy frames. Very blurry or dark images may have reduced accuracy.

**Q: Can I fine-tune the model on my data?**
A: Yes, but requires additional code. Contact for fine-tuning scripts.

**Q: Is real-time inference possible?**
A: With GPU, yes! ~100ms per image allows for near-real-time processing.

---

## 🎯 Use Cases

1. **Clinical Decision Support**: Assist endoscopists during procedures
2. **Quality Assurance**: Validate detection quality post-procedure
3. **Research**: Benchmark tool for polyp segmentation algorithms
4. **Training**: Educational resource for medical imaging students
5. **Annotation**: Semi-automatic annotation tool for datasets

---

## ✅ Checklist for First Run

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `sddeeplab_final.pth` in project directory
- [ ] `Architecture_diagram.png` in project directory
- [ ] Run app command executed (`streamlit run app.py`)
- [ ] Browser opened to http://localhost:8501
- [ ] Model loaded successfully
- [ ] Test image uploaded and segmented

---

**Made with ❤️ for better healthcare** 🏥
