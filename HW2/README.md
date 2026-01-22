# Medical Image Analysis - Homework 2

A comprehensive Jupyter notebook repository for medical image analysis techniques applied to medical imaging datasets. This assignment covers advanced image preprocessing, standardization, and transformation methods essential for deep learning pipelines in medical imaging.

## 📋 Project Overview

This homework assignment focuses on critical preprocessing techniques used in medical image analysis workflows. The three main exercises address fundamental problems in medical imaging:

- **Histogram Normalization**: Standardizing intensity distributions across heterogeneous imaging datasets to reduce confounding factors and improve machine learning model robustness.
- **Bias Field Correction**: Correcting intensity inhomogeneities caused by MRI hardware imperfections, which can degrade image quality and analysis accuracy.
- **Image Resampling**: Converting images to isotropic resolution and aligning coordinate systems for consistent volumetric analysis and visualization.

These preprocessing techniques are essential for preparing raw medical images for downstream tasks such as segmentation, registration, and volumetric measurements.

## 🧠 What I Learned

### Key Concepts
- **Histogram Normalization Principles**: Understanding how percentile-based standardization creates a common intensity profile across image datasets, enabling consistent network behavior across diverse imaging sources.
- **Statistical Landmarks & Interpolation**: Applying linear interpolation to map intensity values based on percentile landmarks, learning the mathematical foundation ($$x' = s_1 + \frac{x - p_{1}}{p_2-p_1}(s_2 - s_1)$$) of intensity mapping.
- **Bias Field Correction Methods**: Learning about MRI bias fields, their causes (hardware imperfections), and correction using the N4 Bias Field Correction algorithm for improved tissue contrast.
- **Image Resampling & Coordinate Systems**: Understanding voxel spacing, isotropic resolution conversion, spatial dimensions, and the importance of maintaining consistent origin/direction information in medical images.
- **Otsu Thresholding**: Applying automated threshold selection methods to create anatomical masks without manual intervention.

### Skills Developed
- Working with medical imaging libraries (nibabel, SimpleITK) for loading, manipulating, and analyzing 3D medical data.
- Image visualization and histogram analysis for quality assessment and preprocessing validation.
- Implementation of mathematical algorithms in NumPy for image transformations.
- Working with both 2D slices (extracted from 3D volumes) and full 3D volumetric data.
- Spatial image operations including masking, filtering, and coordinate system transformations.

## 🛠️ Technologies & Tools Used

### Programming Languages & Frameworks
- **Python 3**: Core programming language for medical image processing and analysis.
- **Jupyter Notebook**: Interactive computing environment for exploratory analysis and reproducible research.

### Libraries & Tools
- **nibabel**: Reading and manipulating NIfTI medical image formats (common in neuroimaging).
- **SimpleITK**: Comprehensive medical image processing toolkit with filters for registration, segmentation, and analysis.
- **NumPy**: Numerical computing and array operations for image data manipulation.
- **SciPy** (`scipy.interpolate`): Advanced scientific computing including interpolation functions for intensity mapping.
- **Matplotlib**: Visualization of images, histograms, and processing results.
- **ITK-SNAP**: 3D image visualization and segmentation tool for validating preprocessing results.

## 📊 Data Description

### Datasets Used

**Hippocampus Dataset (Exercise 1)**
- **Source**: Custom medical imaging dataset (20 3D volumes)
- **Modality**: Structural brain MRI
- **Format**: NIfTI (.nii.gz) compressed format
- **Processing**: Middle axial slices extracted from 3D volumes for 2D analysis
- **Preprocessing**: Intensity normalization to 100-scale maximum
- **Purpose**: Training dataset for histogram normalization algorithm

**Brain MRI with Bias Field (Exercise 2)**
- **Source**: Clinical brain MRI scan with simulated/natural bias field corruption
- **Modality**: T1-weighted or similar structural MRI
- **Format**: NIfTI (.nii.gz)
- **Size**: 3D volume (demonstrates full volumetric processing)
- **Characteristics**: Contains typical MRI intensity inhomogeneities requiring correction
- **Preprocessing**: Otsu thresholding applied to create anatomical masks

**Labeled Brain Anatomy (Exercise 3)**
- **Source**: Patient brain scan with corresponding segmentation mask
- **Format**: Paired NIfTI (.nii.gz) files (image + binary mask)
- **Data**: Non-isotropic spacing requiring standardization
- **Purpose**: Demonstrating clinical workflow for image alignment and coordinate system harmonization

## 📓 Notebooks Description

### Homework2 (1).ipynb

A comprehensive notebook with three integrated exercises demonstrating core medical image preprocessing techniques:

**Exercise 1: Histogram Normalisation (30 points)**
- Implements percentile-based histogram standardization algorithm
- Trains on a multi-subject hippocampus dataset to learn average intensity mapping
- Functions: `calc_landmarks_from_percentiles()`, `create_trained_mapping()`, `standardise_image()`
- Demonstrates before/after histograms and visual validation
- Covers: landmark calculation, interpolation (with extrapolation), averaging mappings across subjects

**Exercise 2: Bias Field Correction (15 points)**
- Corrects MRI intensity inhomogeneities using the N4 Bias Field Correction algorithm
- Implements head masking via Otsu thresholding
- Applies image shrinking for computational efficiency (factor of 4)
- Extracts and visualizes the bias field for quality assessment
- Demonstrates: image rescaling, masking operations, bias estimation, and correction application

**Exercise 3: Image Resampling (15 points)**
- Converts non-isotropic images to 1mm isotropic resolution
- Aligns mask and image coordinate systems for overlay visualization
- Function: `convert_to_isotropic()` for flexible resampling with different interpolation methods
- Validates results by ensuring image and mask dimensions match
- Demonstrates: spatial coordinate transformations, reference image matching, output format saving

## 🚀 How to Use/Run the Notebooks

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Setup Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/[your-username]/Medical-Image-Analysis/HW2.git
   cd HW2
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install required packages:
   ```bash
   pip install notebook jupyter
   pip install numpy scipy matplotlib
   pip install nibabel
   pip install SimpleITK
   ```

4. **Download Required Data Files**
   
   The notebook includes automatic download commands for Homework 2 Hippocampus dataset via `wget`. For other exercises, datasets are referenced via Google Drive links:
   - Exercise 2: Brain MRI image (`img_hw2_q2.nii.gz`)
   - Exercise 3: Brain image and mask (`img_hw2_q3_19676E.nii.gz`, `mask_hw2_q3_19676E.nii.gz`)
   
   Update the file paths in the notebook cells to match your local directory structure.

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook "Homework2_Q(1-3).ipynb"
   ```

6. **Run Cells Sequentially**
   - Start from the top and run each cell in order (Shift+Enter)
   - The notebook includes setup cells that install and import required packages
   - Visualizations (histograms, images) will display inline
   - Expected runtime: 5-15 minutes depending on system performance

### Output Files

After completing all exercises, the notebook generates:
- **resampled_image.nii.gz**: Isotropic resampled brain image
- **resampled_mask.nii.gz**: Resampled segmentation mask aligned with the image
- **Matplotlib figures**: Histograms, overlays, and bias field visualizations displayed inline

### Visualization Tips

- Use **ITK-SNAP** (https://www.itksnap.org/) for interactive 3D visualization of resampled images and masks
- Open `resampled_image.nii.gz` as the main image and `resampled_mask.nii.gz` as an overlay
- Matplotlib histograms provide quantitative validation of normalization effectiveness

## 📝 Notes for Reviewers & Recruiters

This assignment demonstrates proficiency in:
- ✅ Medical image preprocessing and standardization
- ✅ Working with 3D volumetric data and complex spatial transformations
- ✅ Advanced NumPy/SciPy operations for scientific computing
- ✅ Understanding medical imaging formats and coordinate systems
- ✅ Algorithm implementation and mathematical problem-solving
- ✅ Writing reproducible, well-documented code in Jupyter notebooks

The techniques learned here form the foundation of medical image analysis pipelines used in:
- Deep learning-based medical image segmentation
- Image registration and atlas-based analysis
- Volumetric measurements and surgical planning
- Multi-center clinical studies requiring standardized preprocessing

## 📚 References & Further Reading

- **Histogram Normalization**: Reinhold et al. "A new approach to MRI intensity standardization" (IEEE Trans Med Imaging)
- **N4 Bias Field Correction**: Tustison et al. "N4ITK: Improved N3 Bias Correction" (IEEE Trans Med Imaging)
- **SimpleITK Documentation**: https://simpleitk.org/
- **Medical Imaging Fundamentals**: https://en.wikipedia.org/wiki/Medical_imaging

## 💡 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'nibabel'` | Run `pip install nibabel` in your environment |
| `ModuleNotFoundError: No module named 'SimpleITK'` | Run `pip install SimpleITK` in your environment |
| Image file not found | Verify file paths are absolute and datasets are properly downloaded |
| Jupyter kernel crashes on large images | Reduce image dimensions or use shrinking as shown in Exercise 2 |
| Visualization not showing | Ensure `%matplotlib inline` is present and run cells in order |

---

