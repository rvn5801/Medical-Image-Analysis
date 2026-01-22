# Medical Image Analysis - Homework Repository

A comprehensive collection of machine learning and image processing projects for medical imaging. This repository showcases practical implementations of advanced techniques including image preprocessing, deep learning segmentation, classification, and image registration applied to real-world medical imaging datasets.

**Course**: Medical Image Analysis (Spring 2025) | **Institution**: Stony Brook University

---

## 📋 Repository Overview

This repository contains four comprehensive homework assignments demonstrating end-to-end workflows in medical image analysis:

| Assignment | Focus Area | Key Skills | Data |
|---|---|---|---|
| **[Homework 1](#homework-1-medical-image-preprocessing)** | Image Preprocessing | Histogram normalization, bias correction, resampling | Hippocampus CT volumes |
| **[Homework 2](#homework-2-chest-xray-classification)** | Deep Learning Classification | MONAI, transfer learning, class imbalance handling | CheXpert chest X-rays |
| **[Homework 3](#homework-3-3d-artery-segmentation)** | 3D Segmentation | U-Net, patch-based training, vesselness filtering | CT artery datasets |
| **[Homework 4](#homework-4-medical-image-registration)** | Image Registration | ANTs, affine transforms, coordinate transformations | Paired lung CT scans |

---

## 🎓 What I Learned

### Domain Knowledge
- **Medical Image Formats**: NIfTI, DICOM; spatial metadata (spacing, origin, direction)
- **Imaging Modalities**: CT (Hounsfield Units), MRI, X-ray; modality-specific preprocessing
- **Clinical Concepts**: Pleural effusion, consolidation, vascular segmentation, artery propagation
- **Preprocessing Fundamentals**: Normalization, bias correction, registration, coordinate systems

### Machine Learning & Deep Learning
- **Transfer Learning**: Leveraging pretrained models (EfficientNet-B7, ResNet) for medical tasks
- **Segmentation Architectures**: 3D U-Net design, skip connections, encoder-decoder patterns
- **Classification Pipelines**: Multi-class classification, handling imbalanced datasets
- **Data Augmentation**: Domain-appropriate augmentations (rotation, zoom) maintaining clinical validity
- **Loss Functions**: Dice loss for segmentation, cross-entropy for classification
- **Evaluation Metrics**: Dice coefficient, precision-recall curves, AUC-ROC, F1-scores
- **Class Imbalance**: Weighted sampling, stratified splits, balanced vs. unbalanced training analysis

### Advanced Techniques
- **Multi-Resolution Processing**: Hierarchical registration and segmentation for efficiency
- **Patch-Based Training**: Memory-efficient processing of large 3D volumes
- **Sliding Window Inference**: Full-volume predictions from patch-trained models
- **Vesselness Filtering**: Multi-scale Frangi/Objectness filters for vessel enhancement
- **Image Registration**: Affine transformations, Mutual Information metrics, inverse transforms
- **Coordinate Transformations**: Physical vs. pixel space conversions; landmark propagation

### Engineering Skills
- **Jupyter Notebooks**: Interactive development, reproducible research documentation
- **GPU Computing**: CUDA-accelerated training with PyTorch; resource optimization
- **Data Pipelines**: CacheDataset, multi-threaded DataLoaders, efficient I/O
- **Reproducibility**: Deterministic seeding, version control, hyperparameter tracking
- **Visualization**: Medical image slicing, overlay visualization, result validation
- **Hyperparameter Tuning**: Learning rates, batch sizes, augmentation probabilities

---

## 🛠️ Technologies & Tools Used

### Core Programming
- **Python 3.8+** – Primary language for all projects
- **Jupyter Notebook** – Interactive development and documentation

### Deep Learning & ML Libraries
- **PyTorch 2.0+** – Neural network framework with GPU support (CUDA 11.8+)
- **MONAI 1.3.0** – Medical Imaging Open Network for AI
  - Specialized transforms for medical imaging
  - Domain-specific architectures (U-Net, EfficientNet, DenseNet)
  - Medical-imaging metrics (Dice, ROC-AUC)
  - CacheDataset for efficient training
- **scikit-learn** – Machine learning utilities (metrics, sampling, splitting)
- **NumPy 1.23.5** – Numerical computing and array operations
- **SciPy** – Scientific computing (interpolation, signal processing)

### Medical Image Processing
- **SimpleITK** – Image I/O, transforms, resampling, registration
- **nibabel** – NIfTI format handling
- **ANTs (Advanced Normalization Tools)** – Image registration framework
- **ITK-SNAP** – Interactive 3D visualization and annotation

### Visualization & Analysis
- **Matplotlib** – 2D plotting, slice visualization, training curves
- **Pandas** – Data manipulation and analysis (CSV handling, label management)

### Infrastructure & Hardware
- **NVIDIA GPUs** – V100, H100, or equivalent for accelerated training
- **CUDA Toolkit 11.8+** – GPU computing platform
- **Google Colab / Stony Brook SCC** – Cloud and HPC computing environments

---

## 📚 Repository Structure

```
medical-image-analysis-hw/
├── README.md                           # This file
│
├── hw1/                               # Homework 1: Image Preprocessing
│   ├── 116066299_hw1 (1).ipynb
│   └── README.md                      # Detailed assignment breakdown
│
├── hw2/                               # Homework 2: X-ray Classification & Q4 Bonus
│   ├── Homework2 (1).ipynb            # Q1-Q3: Preprocessing fundamentals
│   ├── Homework2_Q4.ipynb             # Q4: MONAI chest X-ray classification
│   ├── README.md                      # Q1-Q3 detailed guide
│   └── README_Q4.md                   # Q4 detailed guide
│
├── hw3/                               # Homework 3: 3D Segmentation
│   ├── hw3_116066299.ipynb            # U-Net segmentation with vesselness filtering
│   └── README.md                      # Detailed assignment breakdown
│
├── hw4/                               # Homework 4: Image Registration
│   ├── hw4_q1.ipynb                   # ANTs registration & transformations
│   └── README.md                      # Detailed assignment breakdown
│
└── requirements.txt                   # Python dependencies
```

---

## 📖 Homework Assignments

### Homework 1: Medical Image Preprocessing

**Notebooks**: [hw1/116066299_hw1 (1).ipynb](hw1/116066299_hw1%20(1).ipynb) | **Guide**: [hw1/README.md](hw1/README.md)

**Objective**: Master fundamental image preprocessing techniques essential for medical imaging pipelines.

**Topics Covered**:
1. **Histogram Normalization** – Standardizing intensity distributions across datasets
   - Percentile-based landmark detection
   - Linear interpolation for intensity mapping
   - Multi-subject averaging for population normalization
   
2. **Bias Field Correction** – Correcting MRI intensity inhomogeneities
   - Otsu thresholding for mask creation
   - N4 Bias Field Correction algorithm
   - Image shrinking for computational efficiency
   
3. **Image Resampling** – Converting to isotropic resolution
   - Spatial coordinate transformations
   - Reference image matching for alignment

**Skills**: NumPy operations, image coordinate systems, statistical image processing, SciPy interpolation

**Dataset**: Hippocampus CT volumes (20 training samples)

---

### Homework 2: Chest X-ray Classification with MONAI

**Notebooks**: 
- [hw2/Homework2_Q(1-3).ipynb](hw2/Homework2_Q(1-3)(1).ipynb) 
- [hw2/Homework2_Q4.ipynb](hw2/Homework2_Q4.ipynb) 

**Guides**: 
- [hw2/README.md](hw2/README.md) (Q1-Q3)
- [hw2/README_Q4.md](hw2/README_Q4.md) (Q4)

**Objective**: Build a production-ready deep learning classifier for chest radiograph interpretation.

**Topics Covered**:
1. **Dataset Splitting & Augmentation** (Exercises 1-3)
   - Train/validation/test split strategies
   - MONAI transform pipelines
   
2. **X-ray Classification** (Exercise 4 )
   - Classifying three conditions: No Finding, Pleural Effusion, Consolidation
   - EfficientNet-B7 architecture with transfer learning
   - Handling class imbalance with weighted sampling
   - Comparison: Balanced vs. unbalanced training approaches

**Skills**: Deep learning with PyTorch, MONAI transforms, GPU training, class imbalance solutions, model evaluation metrics

**Dataset**: CheXpert dataset – 300+ chest X-rays with diagnostic labels (70/10/20 train/val/test split)

**Key Results**: 
- Unbalanced model: ~85% accuracy (89% recall on majority class, 45% on minority)
- Balanced model: ~82% accuracy (87% on majority, 65% on minority class – **20% improvement on rare pathologies**)

---

### Homework 3: 3D Artery Segmentation with MONAI

**Notebook**: [hw3/hw3_116066299.ipynb](hw3/hw3_116066299.ipynb) | **Guide**: [hw3/README.md](hw3/README.md)

**Objective**: Build an end-to-end 3D medical image segmentation pipeline with advanced preprocessing.

**Topics Covered**:
1. **Data Preprocessing for 3D Volumes**
   - Hounsfield Unit normalization ([-1024, 600] range)
   - Foreground cropping and orientation standardization
   - Patch-based training strategy (96³ voxels per patch)
   - Stratified foreground/background sampling

2. **3D U-Net Architecture**
   - Encoder-decoder structure with skip connections
   - Multi-resolution feature extraction
   - Batch normalization and residual units

3. **Advanced Filtering**
   - Vesselness enhancement with multi-scale Gaussian smoothing
   - Objectness Measure filters for vessel detection
   - Dual-channel input (raw + filtered) for improved accuracy

4. **Efficient Inference**
   - Sliding window inference for full-volume predictions
   - Prediction aggregation with 50% overlap
   - Memory-efficient batch processing

**Skills**: 3D deep learning, patch-based processing, SimpleITK filtering, sliding window inference, multi-scale analysis

**Dataset**: 100 CT artery volumes (60 train, 20 val, 20 test)

**Key Results**: 
- Baseline validation Dice: ~0.75
- Enhanced filtering: Potential 2-5% improvement on small vessels

---

### Homework 4: Medical Image Registration with ANTs

**Notebook**: [hw4/hw4_q1.ipynb](hw4/hw4_q1.ipynb) | **Guide**: [hw4/README.md](hw4/README.md)

**Objective**: Master image registration for aligning medical images and propagating segmentations.

**Topics Covered**:
1. **Image Registration Fundamentals**
   - Affine transformations (12-parameter linear alignment)
   - Mutual Information metric for robustness
   - Multi-resolution hierarchical registration

2. **Coordinate System Transformations**
   - Pixel index ↔ Physical coordinate conversions
   - Forward transformations (fixed → moving space)
   - Inverse transformations (moving → fixed space)

3. **Transform Application**
   - Resampling with appropriate interpolators
   - Nearest-neighbor for categorical masks
   - Full-volume inference with registered geometry

4. **Practical Applications**
   - Landmark point propagation
   - Segmentation mask transfer between image spaces
   - ITK-SNAP visualization and validation

**Skills**: ANTs command-line tools, SimpleITK coordinate transformations, image resampling, transform composition

**Dataset**: Paired lung CT scans with expert segmentation masks

**Execution Time**: 5-10 minutes per registration (8-core CPU)

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/rvn5801/Medical-Image-Analysis.git
cd Medical-Image-Analysis
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install jupyter notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai[nibabel,pillow,scikit-image,scikit-learn,scipy]==1.3.0
pip install SimpleITK matplotlib numpy pandas scikit-learn scipy
```

### 4. Download Datasets

Each homework requires datasets from Google Drive or provided links. See individual README files for download instructions:
- **hw1**: [Hippocampus dataset](https://github.com/charlesyou999648/MedIA/tree/main/img)
- **hw2**: [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **hw3**: [CT Artery dataset](https://drive.google.com/drive/folders/1vUdhDu9qLvZ9XEpyl7OV9n_iPmDLGC8i)
- **hw4**: [Lung CT scans](https://drive.google.com/drive/folders/1yCCzXi8fNZ_4kwf7i6q0hAPJO68VFIj5)

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open the desired notebook and run cells sequentially from top to bottom.

---

## 📊 Learning Progression

The homework assignments follow a logical progression:

```
HW1: Preprocessing Basics
  ↓
  Learn: Image formats, intensity normalization, spatial operations
  
HW2: Classification (2D Images)
  ↓
  Learn: Deep learning, transfer learning, class imbalance, MONAI
  
HW3: Segmentation (3D Volumes)
  ↓
  Learn: 3D networks, patch-based processing, advanced filtering
  
HW4: Registration (Spatial Alignment)
  ↓
  Learn: Geometric transformations, coordinate systems, transform composition
```

Each assignment builds on previous skills:
- HW1 preprocessing concepts → HW2/3 input normalization
- HW2 PyTorch/MONAI → HW3 advanced network training
- HW1 coordinate systems → HW4 registration transforms

---

## 💡 Key Takeaways

### Problem-Solving Approaches
1. **Start Simple** → Understand baseline before advanced techniques
2. **Validate Visually** → Always inspect results via slices/overlays
3. **Compare Methods** → Evaluate trade-offs (HW2 balanced vs. unbalanced)
4. **Document Findings** → Record hyperparameters, convergence patterns

### Clinical Relevance
- Preprocessing ensures model robustness across scanners/protocols
- Classification enables automated triage of chest X-rays
- Segmentation identifies anatomical structures for surgery planning
- Registration aligns temporal follow-ups and population studies

### Research-Ready Skills
- Reproducible machine learning workflows
- GPU-accelerated computation
- Domain-specific deep learning frameworks (MONAI)
- Advanced evaluation metrics beyond accuracy
- Proper handling of imbalanced medical data

---

## 📖 Detailed Assignment Guides

For in-depth information about each assignment, refer to the individual README files:

- **[hw1/README.md](hw1/README.md)** – Histogram normalization, bias correction, resampling
- **[hw2/README.md](hw2/README.md)** – Preprocessing exercises overview
- **[hw2/README_Q4.md](hw2/README_Q4.md)** – Chest X-ray classification with MONAI (40 pts)
- **[hw3/README.md](hw3/README.md)** – 3D artery segmentation with U-Net
- **[hw4/README.md](hw4/README.md)** – Image registration with ANTs

---

## 🔧 Troubleshooting

### General Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named X` | Run `pip install X` |
| Out of GPU memory | Reduce batch_size; decrease image resolution |
| Dataset files not found | Verify download paths; check directory structure |
| Notebook kernel crashes | Restart kernel; check system resources with `nvidia-smi` |

### Assignment-Specific

See individual README files for detailed troubleshooting sections:
- **GPU Issues**: See hw2/hw3 READMEs for CUDA setup
- **ANTs Problems**: See hw4/README.md for installation troubleshooting
- **Data Loading**: See hw1/hw3 READMEs for dataset structure validation


---

## 🎯 Learning Outcomes

Upon completing this homework repository, you will be able to:

- [ ] Implement preprocessing pipelines for medical images
- [ ] Build and train deep learning models for medical imaging tasks
- [ ] Handle imbalanced datasets in classification problems
- [ ] Develop 3D segmentation networks with efficient training strategies
- [ ] Perform image registration and coordinate transformations
- [ ] Validate results using appropriate medical imaging metrics
- [ ] Document and communicate technical findings professionally

---

## 🔗 References & Resources

### Official Documentation
- [MONAI Documentation](https://docs.monai.io/)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ANTs Manual](https://github.com/stnava/ANTsDoc/raw/master/ants2.pdf)

### Key Papers
- Ronneberger et al. (2015) – "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Tan & Le (2019) – "EfficientNet: Rethinking Model Scaling for CNNs"
- Maes et al. (1997) – "Multimodality Image Registration by Mutual Information"
- Frangi et al. (1998) – "Multiscale Vessel Enhancement Filtering"

### Learning Resources
- [ITK-SNAP Software](http://www.itksnap.org/) – 3D medical image visualization
- [3D Slicer](https://www.slicer.org/) – Advanced medical image processing
- [Medical Imaging Survey](https://ieeexplore.ieee.org/document/4587903) – Comprehensive overview

---

## 📄 License

This repository is for educational purposes as part of the Medical Image Analysis course at Stony Brook University. Use and modification are permitted for educational and research purposes only.

---

## 🙏 Acknowledgments

- Course instructors for clear problem statements and guidance
- MONAI community for excellent medical imaging deep learning tools
- ANTs developers for powerful registration framework
- Dataset providers (CheXpert, public CT repositories)

---

