# 🚀 Quick Start Guide

Get your 3D segmentation tool running in 5 minutes!

## ⚡ Prerequisites
- Python 3.10+ with Conda/Miniconda
- Internet connection
- Free Modal account

## ⚡ Installation (5 minutes)

### 1. Setup Environment
```bash
# Create and activate environment
conda create -n segment3d python=3.10
conda activate segment3d

# Install dependencies
conda install opencv pyqt
pip install numpy-stl vtk trimesh modal
```

### 2. Setup Modal Cloud
```bash
# Configure Modal (opens browser)
python -m modal setup

# Deploy SAM model (~20 seconds)
modal deploy sam_modal.py
```

### 3. Run Application
```bash
python src/main.py
```

## ⚡ First Test

1. **Select STL File**: Click "Browse" → Choose any .stl file
2. **Open Interface**: Click "Continue" 
3. **Segment**: Right-click on the 3D model
4. **Watch Console**: Look for "Modal segmentation successful!"
5. **View Results**: Red overlay shows segmented region

## ⚡ Expected Output
```bash
Initializing Modal SAM client...
Modal SAM deployment is healthy and ready!
Generating mask using Modal SAM...
Modal segmentation successful! Message: Segmentation successful
```

## ⚡ Troubleshooting

**Import Errors?**
```bash
pip install opencv-python  # if cv2 missing
pip install numpy-stl      # if stl issues
```

**Modal Issues?**
```bash
modal app list             # check deployment
modal deploy sam_modal.py  # re-deploy if needed
```

**Need Test STL?** Download from [Thingiverse](https://www.thingiverse.com) or use any .stl file.

## ⚡ What's Next?

- ✅ Your app is running with cloud AI
- ✅ No local GPU/CUDA needed  
- ✅ ~$0.008 per segmentation
- ✅ 3,750 free segmentations/month

Ready to build the web version? See the full [README.md](README.md) for details!

---
**🎉 You're all set!** Right-click on 3D models to see AI segmentation in action. 