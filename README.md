# Segment3D - AI-Powered 3D Model Segmentation Tool

A sophisticated 3D model segmentation application that uses Meta's Segment Anything Model (SAM) for intelligent region selection and annotation. The application has been enhanced with **cloud-powered AI processing** using Modal for scalable, GPU-accelerated segmentation.

## 🚀 Features

- **3D Model Visualization**: Interactive VTK-based 3D model viewer
- **AI-Powered Segmentation**: Uses SAM (Segment Anything Model) for intelligent region selection
- **Cloud Processing**: SAM model runs on Modal cloud GPUs - no local GPU required
- **Point-Click Interface**: Right-click on 3D models to trigger segmentation
- **STL Export**: Generate segmented STL files from your selections
- **Multi-view Support**: View original and segmented models side-by-side

## 🏗️ Architecture

```
Local Application (PyQt5)     →     Modal Cloud (GPU)     →     Results
┌─────────────────────┐             ┌─────────────────┐         ┌──────────┐
│ • 3D Viewer (VTK)   │  Screenshot │ • SAM Model     │ Masks   │ • Overlay│
│ • User Interaction  │ ────────────→ • T4 GPU       │ ────────→ │ • Export │
│ • File Management   │             │ • Auto-scaling  │         │ • Display│
└─────────────────────┘             └─────────────────┘         └──────────┘
```

## 📋 Prerequisites

- **Python 3.10+** (recommended: Anaconda/Miniconda)
- **Internet Connection** (for Modal cloud processing)
- **Modal Account** (free tier includes $30/month credits)

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd segment-3d
```

### Step 2: Set Up Python Environment
```bash
# Using Conda (recommended)
conda create -n segment3d python=3.10
conda activate segment3d

# Or using venv
python -m venv segment3d
# Windows:
segment3d\Scripts\activate
# Mac/Linux:
source segment3d/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install core packages
conda install opencv
pip install numpy-stl vtk trimesh modal

# For PyQt5 (if not already installed)
conda install pyqt  # or pip install PyQt5
```

### Step 4: Set Up Modal Cloud
```bash
# Install and configure Modal
pip install modal
python -m modal setup
# This will open a browser for authentication
```

### Step 5: Deploy SAM Model to Modal
```bash
modal deploy sam_modal.py
```

You should see:
```
✓ App deployed in ~20s! 🎉
View Deployment: https://modal.com/apps/[your-username]/main/deployed/sam-segmentation
```

## 🚀 Usage

### Running the Application
```bash
python src/main.py
```

### Step-by-Step Workflow

1. **File Selection Window**
   - Click "Browse" to select an STL file
   - Click "Continue" to open the main interface

2. **3D Segmentation Interface**
   - **Rotate/Zoom**: Left-click and drag to rotate, scroll to zoom
   - **Segment**: Right-click on any part of the 3D model
   - **Toggle Views**: Use "Hide Model" to switch between original and segmented views

3. **Processing Flow**
   ```
   Right-click → Screenshot → Modal SAM → Segmentation → Display
   ```

4. **Results**
   - Segmented regions appear in red overlay
   - STL files are saved to `output/` directory
   - Processing logs appear in console

### Expected Console Output
```bash
Initializing Modal SAM client...
Modal SAM deployment is healthy and ready!
Generating mask using Modal SAM for image: temp/screenshot0.jpg
Modal segmentation successful! Message: Segmentation successful
```

## 🎛️ Controls

| Action | Control |
|--------|---------|
| Rotate 3D Model | Left-click + drag |
| Zoom | Mouse wheel |
| Segment Region | Right-click on model |
| Toggle Original/Segmented | "Hide Model" button |
| Start Segmentation | "Segment" button |

## 📁 Project Structure

```
segment-3d/
├── src/
│   ├── main.py                 # Application entry point
│   ├── modal_client.py         # Modal cloud integration
│   └── frame/
│       ├── file_selector.py    # File selection dialog
│       ├── segmentation_tool.py # Main 3D interface
│       └── stl_view.py         # STL viewer dialog
├── sam_modal.py               # Modal SAM deployment
├── test_modal_integration.py  # Integration tests
├── requirements.txt           # Dependencies
├── output/                    # Generated STL files
└── temp/                      # Temporary processing files
```

## 💰 Cost Information

### Modal Pricing (Pay-per-use)
- **GPU Used**: T4 (~$0.0004/second)
- **Per Segmentation**: ~$0.008 (2-3 seconds processing)
- **Monthly Free Tier**: $30 = ~3,750 segmentations
- **Cost Example**: 100 segmentations = $0.80

### Estimated Usage Costs
| Monthly Segmentations | Cost | Within Free Tier |
|----------------------|------|------------------|
| 100 | $0.80 | ✅ Yes |
| 1,000 | $8.00 | ✅ Yes |
| 3,000 | $24.00 | ✅ Yes |
| 5,000 | $40.00 | ❌ $10 overage |

## 🧪 Testing

### Test Modal Integration
```bash
python test_modal_integration.py
```

Expected output:
```
✅ All tests PASSED!
- Modal connection: ✓
- SAM segmentation: ✓ (score: 1.0021)
- Generated test images for inspection
```

### Test with Sample STL
1. Download any STL file (e.g., from Thingiverse)
2. Run the application
3. Load the STL file
4. Right-click different parts to test segmentation

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
conda install opencv
# or
pip install opencv-python
```

**"No module named 'modal'"**
```bash
pip install modal
python -m modal setup
```

**"Modal segmentation failed"**
- Check internet connection
- Verify Modal deployment: `modal app list`
- Re-deploy if needed: `modal deploy sam_modal.py`

**STL Import Issues**
```bash
pip uninstall stl
pip install numpy-stl
```

**VTK/Qt Issues**
```bash
conda install vtk pyqt
# or
pip install vtk PyQt5
```

### Debug Mode
Add this to see detailed logs:
```python
# In segmentation_tool.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔧 Configuration

### Modal Settings
Edit `sam_modal.py` to customize:
- **GPU Type**: Change `gpu="T4"` to `gpu="A10G"` for faster processing
- **Timeout**: Modify `scaledown_window=300` for different idle times
- **Model**: Switch SAM model variants (vit_b, vit_l, vit_h)

### Application Settings
Edit `segmentation_tool.py`:
- **Output Directory**: Change `self.output_dir`
- **Screenshot Quality**: Modify screenshot resolution
- **UI Styling**: Customize button styles and colors

## 🎯 Performance Tips

1. **Faster Segmentation**: Use smaller screenshots for quicker processing
2. **Better Quality**: Click closer to object edges for more precise masks
3. **Batch Processing**: Process multiple points in one session
4. **Cost Optimization**: Process similar objects together

## 🔮 Future Enhancements

- **Web Application**: React + Three.js frontend
- **2D Image Support**: Extend to handle image annotation
- **Collaboration Features**: Multi-user annotation
- **Export Formats**: COCO, YOLO, Pascal VOC support
- **API Integration**: RESTful API for external tools

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Modal logs: https://modal.com/logs
3. Test the integration: `python test_modal_integration.py`

---

**🎉 Congratulations!** You now have a production-ready 3D segmentation tool powered by cloud AI!