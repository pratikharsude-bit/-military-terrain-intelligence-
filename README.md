# -military-terrain-intelligence-


# Military Terrain Analysis

A deep-learning-based satellite and aerial imagery analysis system that combines image classification and semantic segmentation to identify terrain categories, land-cover patterns, and building structures.

The project integrates multiple computer vision models into an interactive **Streamlit web application**, allowing users to upload satellite or aerial images and view model predictions through visual outputs.

> **Project Type:** Academic / Research Prototype  
> **Technology:** Deep Learning, Computer Vision, Python, Streamlit

---

## Overview

Military Terrain Analysis is designed to explore how deep learning can be applied to satellite imagery for terrain and land-cover analysis.

The system uses multiple trained models, each designed for a specific computer vision task. Instead of relying on a single model, the application combines classification and segmentation approaches to analyse different aspects of an input image.

The application provides an interactive interface where users can upload supported imagery and view the corresponding predictions.

## Key Features

- **Terrain Scene Classification:** Identifies broad aerial scene categories.
- **Land-Cover Segmentation:** Segments images into different land-cover classes.
- **Multispectral Image Classification:** Processes supported 13-band satellite TIFF imagery.
- **Building Detection:** Generates experimental building-footprint segmentation outputs.
- **Visual Analysis:** Displays prediction masks, overlays, confidence scores, and class-wise percentages where supported.
- **Interactive Dashboard:** Provides a user-friendly Streamlit interface for exploring model outputs.

---

## Models and Datasets

The project integrates models trained on multiple publicly available remote-sensing datasets.

| Dataset | Model | Task | Evaluation |
|---|---|---|---|
| SkyView | ResNet-18 | Aerial scene classification | 96.42% Test Accuracy |
| EuroSAT All-Bands | 13-Channel ResNet-18 | Multispectral image classification | 98.26% Evaluation Accuracy |
| DeepGlobe Land Cover | U-Net with ResNet-18 Encoder | Semantic segmentation | 64.85% Validation mIoU |
| SpaceNet-7 | U-Net | Building-footprint segmentation | Experimental |

### 1. SkyView — Terrain Scene Classification

- Architecture: ResNet-18
- Dataset: SkyView Aerial Landscape Dataset
- Classes: 15 aerial scene categories
- Reported test accuracy: **96.42%**

The model predicts the most likely scene category and provides confidence scores and top predictions.

### 2. EuroSAT — Multispectral Classification

- Architecture: 13-channel ResNet-18
- Dataset: EuroSAT All-Bands
- Input: Supported 13-band TIFF imagery
- Reported evaluation accuracy: **98.26% (2,711 / 2,759 samples)**

This model is designed to classify multispectral satellite imagery using information from multiple spectral bands.

### 3. DeepGlobe — Land-Cover Segmentation

- Architecture: U-Net with ResNet-18 encoder
- Dataset: DeepGlobe Land Cover Classification Dataset
- Task: Pixel-level semantic segmentation
- Validation mIoU: **64.85%**
- Validation pixel accuracy: **83.38%**

The model identifies six broad land-cover categories:

- Urban Land
- Agriculture Land
- Rangeland
- Forest Land
- Water
- Barren Land

The application displays segmentation masks, image overlays, and predicted class-wise pixel percentages.

### 4. SpaceNet-7 — Building-Footprint Segmentation

- Architecture: U-Net
- Dataset: SpaceNet-7
- Task: Building-footprint segmentation

This model generates predicted building masks and overlays.

**Note:** Building segmentation is experimental and has shown limited validation performance. Its outputs should be treated as approximate demonstrations rather than reliable building detections.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | ResNet-18, U-Net |
| Web Interface | Streamlit |
| Image Processing | Pillow, Rasterio |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning | scikit-learn |

---

## Project Structure

```text
military-terrain-analysis/
│
├── app/
│   ├── main.py
│   ├── pipeline.py
│   └── inference/
│       ├── skyview_inference.py
│       ├── deepglobe_inference.py
│       ├── eurosat_inference.py
│       └── spacenet_inference.py
│
├── models/
│   └── checkpoints/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
```

*The structure above represents the main application components. Additional training scripts, configuration files, and supporting folders may also be present.*

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/military-terrain-analysis.git
cd military-terrain-analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Model Checkpoints

Model checkpoint files are excluded from this repository because of their size.

Place the required model files inside:

```text
models/checkpoints/
```

Expected filenames:

```text
skyview_resnet18_best.pth
deepglobe_unet_best.pth
eurosat_allbands_resnet18_best.pth
spacenet_unet_best.pth
```

The corresponding model checkpoints and inference modules must be available for the respective predictions to work.

### 5. Run the Application

From the project root directory:

```bash
python -m streamlit run app/main.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## Input Support

- RGB satellite and aerial images are processed by the applicable RGB inference models.
- Supported 13-band TIFF imagery is routed to the EuroSAT multispectral classification model.
- TIFF processing depends on the number of available bands and readable raster data.
- Prediction availability depends on the corresponding model checkpoint and inference module.

---

## Evaluation and Limitations

The reported metrics are based on evaluations performed on the respective datasets.

- SkyView: 96.42% test accuracy.
- EuroSAT All-Bands: 98.26% evaluation accuracy.
- DeepGlobe: 64.85% validation mIoU and 83.38% validation pixel accuracy.
- SpaceNet-7: Experimental building segmentation with limited validation performance.

These metrics are task-specific and are not directly comparable. Performance on uploaded images may vary depending on image resolution, sensor characteristics, geographic region, and differences from the training data.

The project is an academic prototype. It does not provide certified terrain traversability, tactical suitability, or safety assessments. Predictions should always be independently verified.

---

## Future Improvements

- Improve model generalisation across different satellite imagery sources.
- Improve building-footprint segmentation performance.
- Expand evaluation using independent satellite imagery.
- Improve input validation and error handling.
- Add more remote-sensing datasets and classification categories.
- Improve visualization and reporting of model predictions.

---

## Project Purpose

This project demonstrates the application of deep learning and computer vision techniques to satellite imagery analysis through the integration of multiple models into a unified interactive application.

It is developed for academic learning, experimentation, and demonstration purposes.

---

## Responsible Use

This software is intended for educational and research purposes only. It is not a certified military or operational decision-support system. Predictions may be inaccurate and must not be used as the sole basis for safety-critical, navigation, emergency-response, or operational decisions.

---

## License

No open-source license has currently been specified. Please check the original dataset and model licenses before redistributing their data, weights, or derived assets.
