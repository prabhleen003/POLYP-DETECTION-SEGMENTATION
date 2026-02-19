# 📦 Installation & Setup Guide

## System Requirements

### Minimum
- **OS**: Windows 10/11, macOS 10.13+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 4GB
- **Disk Space**: 2GB (for dependencies + model)

### Recommended
- **OS**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or 3.11
- **RAM**: 8GB
- **GPU**: NVIDIA GPU with 2GB+ VRAM and CUDA 11.0+
- **Disk Space**: 4GB

---

## Step-by-Step Installation

### 1. Install Python

#### **Windows**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```bash
   python --version
   ```

#### **macOS**
```bash
# Using Homebrew (recommended)
brew install python@3.10

# Or download from python.org
python3 --version
```

#### **Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3 --version
```

---

### 2. Navigate to Project Directory

```bash
# Windows (PowerShell or CMD)
cd "C:\Users\YourUsername\OneDrive\Desktop\Colorectal polyp segmentatiom"

# macOS/Linux
cd ~/Desktop/"Colorectal polyp segmentatiom"
```

---

### 3. Create Virtual Environment

#### **Option A: Quick Start (Windows)**
Just double-click `run_app.bat` - it will handle everything!

#### **Option B: Manual Setup**

**Windows (Command Prompt)**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell)**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell shows "execution policy" error:
> ```bash
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Upgrade pip

```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installation Time**: 5-15 minutes (depending on internet speed)

#### What Gets Installed:
- PyTorch (core deep learning framework)
- Streamlit (web app framework)
- OpenCV (image processing)
- NumPy, SciPy, scikit-learn (numerical computing)
- And 7 other supporting libraries

---

### 6. GPU Setup (Optional but Recommended)

#### **NVIDIA GPU Users**

**Check your GPU:**
```bash
nvidia-smi
```

If the command works, you have CUDA-capable GPU!

**Install CUDA-optimized PyTorch:**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

#### **Mac or CPU-only Users**
Skip this step - CPU inference works fine (just slower)

---

### 7. Verify Installation

```bash
python -c "
import torch
import streamlit
import cv2
import numpy as np
print('✅ All dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
```

---

### 8. Check Project Files

Make sure you have these files:
```
Colorectal polyp segmentatiom/
├── app.py                          ✓
├── model.py                        ✓
├── utils.py                        ✓
├── requirements.txt                ✓
├── sddeeplab_final.pth            ✓ (IMPORTANT!)
├── Architecture_diagram.png        ✓
├── .streamlit/config.toml         ✓
├── run_app.bat                    ✓ (Windows only)
└── README.md                      ✓
```

If any file is missing, redownload from the project source.

---

## Running the App

### **Windows (Easiest)**
Simply double-click `run_app.bat` and the app will start automatically!

### **Windows (Manual)**
```bash
venv\Scripts\activate
streamlit run app.py
```

### **macOS/Linux**
```bash
source venv/bin/activate
streamlit run app.py
```

### **Expected Output**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

The browser will automatically open. If not, go to: **http://localhost:8501**

---

## Troubleshooting Installation

### Python Not Found
**Error**: `'python' is not recognized as an internal or external command`

**Solution**:
1. Reinstall Python and check "Add Python to PATH"
2. Or use `python3` instead of `python`
3. Or use full path: `C:\Python311\python.exe`

---

### pip Install Fails
**Error**: `ERROR: Could not find a version that satisfies...`

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Try installing again
pip install -r requirements.txt

# If still failing, install packages one by one
pip install streamlit
pip install torch
pip install torchvision
# ... etc
```

---

### Virtual Environment Not Activating
**Error**: Script not found or activation fails

**Solution**:
```bash
# Delete venv folder
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux

# Recreate
python -m venv venv

# Windows Command Prompt (use this!)
venv\Scripts\activate

# Or Windows PowerShell
.\venv\Scripts\Activate.ps1
```

---

### Model Won't Load
**Error**: `Error loading model: No such file or directory`

**Solution**:
1. Verify `sddeeplab_final.pth` is in the project folder
2. Check file is not corrupted (size should be > 100MB)
3. Copy-paste file from backup if needed

---

### Out of Memory
**Error**: `CUDA out of memory` or `MemoryError`

**Solutions**:
1. Use CPU mode:
   ```bash
   CUDA_VISIBLE_DEVICES="" streamlit run app.py  # macOS/Linux
   set CUDA_VISIBLE_DEVICES= && streamlit run app.py  # Windows
   ```

2. Restart the app:
   - Stop with Ctrl+C
   - Wait 10 seconds
   - Run again

3. Close other applications using GPU

---

### Streamlit Not Found
**Error**: `No module named streamlit`

**Solution**:
```bash
# Make sure venv is activated
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then install
pip install streamlit
```

---

### Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process using port 8501
# Windows (PowerShell)
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess -Force

# macOS/Linux
lsof -ti:8501 | xargs kill -9
```

---

## Updating Dependencies

If you need to update packages later:

```bash
# Activate venv first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Update all packages
pip install --upgrade -r requirements.txt

# Or update specific package
pip install --upgrade torch
```

---

## Deactivating Virtual Environment

When done working:

```bash
# Windows / macOS / Linux
deactivate
```

---

## Fresh Installation (if needed)

If you want to start completely fresh:

```bash
# 1. Delete virtual environment
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux

# 2. Create new one
python -m venv venv

# 3. Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 4. Install fresh
pip install -r requirements.txt

# 5. Run
streamlit run app.py
```

---

## Advanced Configuration

### Using Conda (Alternative)

If you prefer Conda:

```bash
# Create environment
conda create -n sddeeplab python=3.10

# Activate
conda activate sddeeplab

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Running on Different Machine

```bash
# Access from another computer on same network
streamlit run app.py --server.address 0.0.0.0

# Then visit: http://your-machine-ip:8501
```

---

## Docker Support (Enterprise)

If deploying on servers, Docker setup available on request.

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed successfully
- [ ] Model file `sddeeplab_final.pth` present (>100MB)
- [ ] Architecture diagram present
- [ ] Streamlit runs without errors
- [ ] App opens in browser
- [ ] Model loads successfully (yellow spinner → green checkmark)
- [ ] Can upload and segment test image

---

## 🆘 Still Having Issues?

1. **Check Python version**: `python --version` (needs 3.8+)
2. **Check pip**: `pip --version`
3. **Verify files**: All 8 files present in project folder
4. **Check disk space**: At least 2GB free
5. **Antivirus**: Sometimes blocks PyTorch downloads. Temporarily disable if needed.
6. **Proxy/Firewall**: If behind corporate proxy, may need configuration

---

**Installation complete! Now run the app and segment some polyps! 🎉**
