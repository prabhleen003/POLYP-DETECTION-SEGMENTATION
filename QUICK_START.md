# 🚀 QUICK START GUIDE

## 30-Second Setup (Windows)

1. **Double-click**: `run_app.bat`
2. **Wait**: Loading prompt appears
3. **Done**: App opens automatically at `http://localhost:8501`

That's it! No command line needed.

---

## Setup with Command Line

### Windows (Command Prompt or PowerShell)
```bash
cd "Colorectal polyp segmentatiom"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
cd Colorectal\ polyp\ segmentatiom
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## First Time Using the App?

### 1️⃣ Load Model
- Sidebar → "Load Model" button
- Wait ~30 seconds, then ✅ appears

### 2️⃣ Upload Image
- Segmentation tab → "Upload Image"
- Select JPG/PNG polyp image

### 3️⃣ Run Segmentation
- Click "Run Segmentation" button
- Results show in 1-2 seconds

### 4️⃣ View Results
- See original, mask, and overlay
- See bounding boxes detected
- Download results if needed

---

## Common Issues & Quick Fixes

| Issue | Fix |
|-------|-----|
| `python not found` | Use `python3` or reinstall Python with PATH |
| `pip not found` | Try `python -m pip install` |
| Model won't load | Check file `sddeeplab_final.pth` exists (>100MB) |
| Slow inference | Use GPU (CUDA) or restart app |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## System Requirements

✅ Minimum: Python 3.8+, 4GB RAM
✅ Better: Python 3.10+, 8GB RAM, GPU
✅ Works on: Windows, macOS, Linux

---

## Folder Structure (After Setup)

```
Colorectal polyp segmentatiom/
├── app.py                          # Main app
├── model.py                        # Model architecture
├── utils.py                        # Helper functions
├── requirements.txt                # Dependencies
├── sddeeplab_final.pth            # Model weights
├── Architecture_diagram.png        # Diagram
├── run_app.bat                    # Windows launcher
├── check_system.py                # System checker
├── .streamlit/config.toml         # Settings
├── INSTALLATION.md                # Full guide
├── QUICK_START.md                 # This file
└── README.md                      # Full documentation
```

---

## What Each Button Does

### Left Sidebar
- **Load Model**: Initialize the model (~30s)
- **Segmentation Threshold**: Adjust sensitivity (0.3-0.9)
- **Show Segmentation Overlay**: Toggle overlay display
- **Show Bounding Boxes**: Toggle bbox display
- **Overlay Transparency**: Adjust opacity

### Segmentation Tab
- **Upload Image**: Select polyp image (JPG/PNG/BMP/TIFF)
- **Run Segmentation**: Execute inference

### Dataset Metrics Tab
- View Kvasir-SEG and CVC-ClinicDB results
- Shows Dice, IoU, Precision, Recall, etc.

### Architecture Tab
- Understand model components
- See data flow pipeline
- Learn about SCCR, ASA, SDAA attention

### About Tab
- Project info
- Benchmark results
- Use cases and applications

---

## Understanding Results

**Metrics shown:**
- **Inference Time**: How long segmentation took (ms)
- **Polyps Detected**: Number of detected polyps
- **Total Polyp Area**: Total pixels of detected polyps
- **Avg Polyp Size**: Average polyp size

**Bounding Boxes:**
- ID, X, Y coordinates  
- Width and Height
- Area in pixels²

---

## Tips for Best Results

✨ **Good Images For:**
- Clear endoscopy frames
- Good lighting
- Polyps of various sizes
- Different shapes and colors

❌ **Avoid:**
- Blurry images
- Very dark frames
- Over-exposed shots
- Non-endoscopy images

---

## Running on GPU vs CPU

**Check if GPU available:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `True` → GPU is ready ✅ (80-150ms per image)
If `False` → Using CPU ⚠️ (300-500ms per image)

---

## Troubleshooting Command Reference

```bash
# Check Python
python --version

# Check installed packages
pip list

# Force reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Run system check
python check_system.py

# Run app on CPU only
set CUDA_VISIBLE_DEVICES= && streamlit run app.py

# Run app with more verbose output
streamlit run app.py --logger.level=debug
```

---

## Next Steps

1. ✅ Run the app
2. ✅ Load the model
3. ✅ Try segmenting a test image
4. ✅ Explore dataset metrics
5. ✅ Check architecture explanation
6. ✅ Download some results

---

## Need More Help?

📖 **Full Installation Guide**: See `INSTALLATION.md`

📚 **Complete Documentation**: See `README.md`

🔧 **System Check**: Run `python check_system.py`

---

**Ready to go? Run the app and segment some polyps! 🔬🎉**
