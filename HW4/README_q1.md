# Medical Image Registration with ANTs

A comprehensive guide to automated medical image registration using Advanced Normalization Tools (ANTs). This homework demonstrates practical applications of image alignment, coordinate system transformations, and mask propagation in medical imaging workflows.

## 📋 Project Overview

This exercise focuses on **image registration** – a fundamental preprocessing step in medical imaging that involves aligning two or more images to a common spatial coordinate system. The task addresses critical real-world problems:

**Problem Statement**: In clinical practice, we often need to:
- Align patient scans acquired at different time points (longitudinal studies)
- Compare images from different modalities to the same anatomical space
- Propagate expert annotations (segmentation masks) between patient images
- Enable statistical analysis across image populations in standardized space

**Approach**: Using 3D lung CT scans, we develop and validate:
- **Image Registration** with Affine transformations to align moving image to fixed image
- **Point Transformation** between fixed and moving image spaces using learned transforms
- **Mask Propagation** – transferring expert segmentations using geometric transformations
- **Multi-resolution Strategy** for computational efficiency and accurate convergence

**Dataset**: Paired lung CT scans with corresponding segmentation masks
- Fixed image: Reference anatomical space
- Moving image: Patient scan to be aligned
- Segmentation masks: Expert-annotated lungs requiring transformation

## 🧠 What I Learned

### Key Concepts

- **Image Registration Fundamentals**: Understanding the problem of finding optimal spatial alignment between images through transformation estimation.

- **Affine Transformations**: Learning 3D affine matrices (12 degrees of freedom) for rotation, translation, scaling, and shearing. Linear but flexible enough for inter-patient alignment.

- **Mutual Information (MI) Metric**: Understanding MI as a similarity measure robust to intensity variations, unlike Sum of Squared Differences (SSD). MI measures statistical dependency between image intensities.

- **Multi-Resolution Registration Strategy**: 
  - Coarse level: Low resolution (4× downsampling) for fast global alignment
  - Intermediate level: 2× downsampling for refinement
  - Fine level: Full resolution for accurate local details
  - Speeds up convergence and improves robustness to local minima

- **Gradient Descent Optimization**: Understanding how gradient step size (0.01) balances convergence speed vs. stability.

- **Coordinate System Transformations**: 
  - **Pixel Index**: Integer array coordinates in image space
  - **Physical Point**: Real-world coordinates in mm/cm using image spacing and origin
  - Registration operates in physical space, requiring bidirectional conversions
  - Critical distinction for accurate landmark propagation

- **Transform Composition**: Applying transforms bidirectionally (fixed→moving, moving→fixed) and understanding inverse transforms for reverse mappings.

- **Segmentation Mask Propagation**: Using nearest-neighbor interpolation to preserve categorical label integrity when applying transforms (vs. linear interpolation for continuous images).

### Skills Developed

- Installing and configuring Advanced Normalization Tools (ANTs)
- Constructing ANTs command-line registration pipelines with proper parameter specification
- Understanding and tuning multi-resolution registration parameters (convergence iterations, smoothing, downsampling factors)
- Converting between pixel indices and physical coordinates using SimpleITK
- Reading and applying affine transformation matrices to points
- Computing and applying inverse transforms
- Resampling images with different interpolation strategies
- Validating registration results using overlay visualization in ITK-SNAP

## 🛠️ Technologies & Tools Used

### Programming Languages & Frameworks
- **Python 3** – Scripting and automation
- **Jupyter Notebook** – Interactive development and documentation

### Medical Image Processing Libraries
- **SimpleITK** – Core library for image I/O, resampling, and coordinate transformations
  - `ReadImage()`, `WriteImage()` – NIfTI format handling
  - `TransformIndexToPhysicalPoint()`, `TransformPhysicalPointToIndex()` – Coordinate conversions
  - `ReadTransform()`, `GetInverse()` – Affine transform management
  - `ResampleImageFilter()` – Image resampling with custom interpolators
  - `sitkNearestNeighbor`, `sitkLinear` – Interpolation modes

- **Advanced Normalization Tools (ANTs)** – Command-line registration framework
  - `antsRegistration` – Main registration executable
  - `antsApplyTransforms` – Transform application to new images
  - Multi-resolution hierarchical registration
  - Mutual Information metric implementation

### Visualization & Analysis
- **Matplotlib** – In-notebook visualization of slices and landmarks
- **ITK-SNAP** – Interactive 3D visualization and overlay inspection
  - Slice-by-slice registration validation
  - 50% transparency overlay for visual assessment
  - Screenshot documentation of results

### Computational Infrastructure
- **Multi-core CPU Processing** – ANTs parallelization (8 cores: ~10 min execution)
- **Operating Systems**: macOS, Linux, Windows (with appropriate ANTs binary)

## 📊 Data Description

### CT Lung Dataset

**Image Characteristics**:
- **Modality**: Computed Tomography (CT) – volumetric 3D imaging
- **Anatomical Region**: Thorax (chest) with focus on lung anatomy
- **Voxel Spacing**: Variable (typically 1mm isotropic)
- **Image Size**: ~512×512×~300 voxels (approximately 75-150 MB per volume)
- **Intensity Range**: Hounsfield Units (HU) ranging from -1024 (air) to +1000 (bone)

**Data Composition**:
- **Fixed Image** (`hw4_fixed_img_14303S.nii`) – Reference anatomical space
- **Moving Image** (`hw4_moving_img_10009Y.nii`) – Patient scan requiring alignment
- **Moving Segmentation Mask** (`hw4_moving_mask_10009Y.nii.gz`) – Expert-annotated lung labels

**Data Processing**:
1. **Format**: NIfTI (.nii, .nii.gz) – preserves spatial metadata (spacing, origin, direction)
2. **Loading**: SimpleITK `ReadImage()` maintains full spatial metadata
3. **Registration Preprocessing**: 
   - Images read with preserved spacing/origin/direction
   - No explicit preprocessing (registration algorithm handles intensity variation via MI)
   - Multi-resolution downsampling handled internally by ANTs

**Mask Characteristics**:
- **Type**: Binary/categorical (0=background, 1=lung tissue)
- **Transformation Strategy**: Requires nearest-neighbor interpolation to preserve categorical labels
- **Propagation**: Applied using learned affine transform from image registration

## 📓 Notebooks Description

### hw4_q1.ipynb

A complete practical guide to medical image registration from preprocessing through validation:

#### **Part 1: Image Registration with ANTs (Task 1.1)**

**Registration Pipeline Setup**
- ANTs binary installation and path configuration
- Fixed and moving image loading with metadata preservation
- Output path specification for transformed images and transforms

**Registration Command Construction**
```
antsRegistration -d 3 \
  -o [output_prefix, output_image] \
  -r [fixed_image, moving_image, 1] \
  -t Affine[0.01] \
  -m MI[fixed, moving, 1, 32, Regular, 0.5] \
  -c [500x250x100] \
  -s 2x1x0 \
  -f 4x2x1
```

**Key Parameters**:
- **Transform Type**: Affine (12-parameter linear transformation, gradient step 0.01)
- **Metric**: Mutual Information with:
  - Radius: 1 voxel neighborhood for intensity correlation
  - Histogram bins: 32 for statistical estimation
  - Sampling strategy: Regular grid
  - Sampling percentage: 50% voxels
- **Multi-resolution Strategy**:
  - Convergence iterations: [500, 250, 100] for [coarse, intermediate, fine]
  - Smoothing: [2, 1, 0] voxels
  - Downsampling factors: [4, 2, 1]
- **Output**: Warped image + affine transform matrix file

**Execution**: Typical runtime 5-10 minutes on 8-core CPU

#### **Part 2: Visualization (Task 1.2)**

- Load warped moving image and fixed image
- Overlay with 50% transparency in ITK-SNAP
- Visual verification of registration quality
- Screenshot documentation of result

#### **Part 3: Fixed-to-Moving Point Transformation (Tasks 1.3.1)**

**Three-Step Transformation Pipeline**:
1. **Pixel-to-Physical**: Convert pixel index on fixed image to physical coordinates
   ```python
   physical_cor_on_fixed = fixed_img.TransformIndexToPhysicalPoint(pixel_index)
   ```
   Uses image spacing, origin, and direction to compute mm-space coordinates

2. **Apply Affine Transform**: Transform physical point from fixed space to moving space
   ```python
   physical_cor_on_moving = affine_transform.TransformPoint(physical_cor_on_fixed)
   ```
   Applies 3×4 affine matrix transformation

3. **Physical-to-Pixel**: Convert moving image physical coordinates back to pixel indices
   ```python
   pixel_on_moving = moving_img.TransformPhysicalPointToIndex(physical_cor_on_moving)
   ```
   Inverse of step 1 using moving image's geometry

**Validation**: Plot target point on both fixed and moving images to verify alignment

#### **Part 4: Moving-to-Fixed Point Transformation (Tasks 1.3.2)**

**Four-Step Reverse Transformation**:
1. **Get Inverse Transform**:
   ```python
   affine_transform_inv = affine_transform.GetInverse()
   ```
   Computes matrix inverse for reverse mapping

2. **Pixel-to-Physical**: Convert moving image pixel to physical coordinates

3. **Apply Inverse Transform**: Transform from moving to fixed physical space

4. **Physical-to-Pixel**: Convert fixed image physical coordinates to pixels

**Use Case**: Identifying anatomical landmarks on moving image and locating them on fixed image

#### **Part 5: Mask Propagation with SimpleITK (Task 1.4)**

**Resampling Function Implementation**:
```python
def resample(image, reference_image, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Preserve categorical labels
    resampler.SetReferenceImage(reference_image)        # Match fixed image space
    resampler.SetTransform(transform)                    # Apply affine transform
    resampler.SetDefaultPixelValue(0)                   # Background label
    return resampler.Execute(image)
```

**Key Decisions**:
- **Nearest Neighbor Interpolation**: Preserves discrete label values (0, 1) without blurring
- **Reference Image**: Ensures output aligns with fixed image grid (spacing, size, origin)
- **Default Value**: Set to 0 (background) for regions outside input image bounds
- **Application**: Transfers moving mask to fixed image space using learned transformation

**Validation**: Overlay warped mask on fixed image to verify anatomical alignment

#### **Part 6: Mask Propagation with antsApplyTransforms (Task 1.5)**

**Command-Line Transform Application**:
```bash
antsApplyTransforms -d 3 \
  -i moving_mask.nii.gz \
  -r fixed_image.nii \
  -o warped_mask.nii.gz \
  -t transform_0GenericAffine.mat \
  -n NearestNeighbor \
  -f 0
```

**Parameters**:
- `-d 3`: 3D operation
- `-i`: Input (moving mask)
- `-r`: Reference image (fixed image space)
- `-o`: Output warped mask
- `-t`: Transformation file
- `-n NearestNeighbor`: Interpolation for categorical data
- `-f 0`: Default fill value

**Equivalent to Part 5** but via command-line, useful for batch processing

#### **Part 7: Final Validation (Task 1.6)**

- Load warped moving image and warped mask
- Overlay mask on warped image in ITK-SNAP
- Visual confirmation of mask-image alignment post-registration
- Screenshot for documentation

#### **Utilities & Helper Functions**

- `pixel_to_point()`: Converts pixel index to spatial coordinates for visualization
- `plot_points_on_grid()`: Overlays landmark points on 3D image slices
- Visualization at arbitrary slice locations for verification

## 🚀 How to Use/Run the Notebook

### Prerequisites

- Python 3.8+
- ANTs 2.5.1 or later installed
- SimpleITK (Python bindings)
- ITK-SNAP for visualization (optional but recommended)
- ~5-10 minutes for registration execution (8-core CPU)

### Environment Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/rvn5801/Medical-Image-Analysis/HW4.git
   cd HW4
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install jupyter notebook
   pip install SimpleITK
   pip install numpy matplotlib
   ```

4. **Install ANTs**

   **Option A: Pre-compiled Binaries** (Recommended)
   
   Download from [ANTs releases](https://github.com/ANTsX/ANTs/releases/download/v2.5.1/):
   - macOS: `ants-2.5.1-arm-macosx-X64.zip` (Apple Silicon) or `ants-2.5.1-macosx-X64.zip` (Intel)
   - Linux: `ants-2.5.1-centos7-X64-gcc.zip`
   - Windows: `ants-2.5.1-windows-X64-gcc.zip`

   Extract and update path in notebook:
   ```python
   ants_path = "/path/to/ants-2.5.1/bin/"
   ```

   **Option B: Compile from Source** (Advanced)
   ```bash
   git clone https://github.com/ANTsX/ANTs.git
   cd ANTs
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j 8
   ```

5. **Download Dataset**
   
   Download from [Google Drive folder](https://drive.google.com/drive/folders/1yCCzXi8fNZ_4kwf7i6q0hAPJO68VFIj5?usp=sharing):
   - `hw4_fixed_img_14303S.nii`
   - `hw4_moving_img_10009Y.nii`
   - `hw4_moving_mask_10009Y.nii.gz`

   Place in working directory and update paths in notebook cells:
   ```python
   fixed_img_path = '/path/to/hw4_fixed_img_14303S.nii'
   moving_img_path = '/path/to/hw4_moving_img_10009Y.nii'
   moving_mask_path = '/path/to/hw4_moving_mask_10009Y.nii.gz'
   ```

6. **Launch Jupyter**
   ```bash
   jupyter notebook hw4_q1.ipynb
   ```

7. **Run Cells Sequentially**
   - Execute cells in order from top to bottom (Shift+Enter)
   - Cell 3 contains main registration command – this will run for 5-10 minutes
   - Allow registration to complete fully before proceeding
   - Verify intermediate outputs exist before running transform application cells
   - Use visualizations to validate results at each step

### Execution Workflow

| Step | Cell # | Duration | Output |
|------|--------|----------|--------|
| Setup & imports | 2 | <1 min | None |
| Registration command | 3 | 5-10 min | Warped image + transform file |
| Visualization (optional) | 5 | <1 min | Screenshot |
| Point transform (fixed→moving) | 8-11 | <1 min | Pixel coordinate mappings |
| Point transform (moving→fixed) | 13-16 | <1 min | Pixel coordinate mappings |
| Mask resampling (SimpleITK) | 18 | 1-2 min | Warped mask |
| Mask transform (ANTs) | 20 | 2-3 min | Warped mask (alternative method) |
| Final validation (optional) | 21 | <1 min | Screenshot |
| **Total** | | **10-20 min** | Complete registration & masks |

### Expected Output Files

1. **Image Outputs**
   - `hw4_warped_img_10009Y.nii.gz` – Moving image aligned to fixed space
   - `hw4_warped_mask_10009Y.nii.gz` – Moving segmentation mask in fixed space (two versions possible)

2. **Transform Matrices**
   - `hw4_warped_img_10009Y_0GenericAffine.mat` – 3×4 affine matrix file (text format)
   - Contains 12 parameters: 3×3 rotation/scale + 3×1 translation

3. **Console Output**
   - ANTs registration progress and convergence information
   - Point transformation coordinates (physical and pixel spaces)
   - Registration metrics and optimization iterations

### Troubleshooting Setup Issues

| Issue | Solution |
|-------|----------|
| `antsRegistration: command not found` | Update `ants_path` variable; verify ANTs installation |
| `SimpleITK not found` | Run `pip install SimpleITK` |
| `Input file not found` | Check paths match actual file locations; update working directory |
| Registration takes >15 minutes | Verify 8+ CPU cores available; check system load with `top`/`Activity Monitor` |
| Transform matrix file not created | Check fixed/moving image paths are correct; verify file formats are NIfTI |

## 📊 Expected Results & Insights

### Registration Quality Metrics
- **Convergence**: Algorithm should converge by final iteration level
- **Visual Alignment**: Moving image should overlay well with fixed image
- **Anatomical Correctness**: Landmark points should map correctly between images

### Point Transformation Validation
- **Forward Transform**: Target point on fixed → corresponding point on moving image
- **Reverse Transform**: Same target point on moving → should return to original location on fixed
- **Consistency**: Inverse operations should recover original coordinates (within numerical precision)

### Mask Alignment
- **Segmentation Overlay**: Moving mask should overlay correctly on warped image
- **Two Methods Match**: SimpleITK resampling and antsApplyTransforms should produce identical results
- **Boundary Preservation**: Mask boundaries should be sharp (no blurring from interpolation)

## 💡 Clinical & Practical Significance

- **Longitudinal Studies**: Align follow-up scans to baseline for disease progression monitoring
- **Atlas-Based Analysis**: Register individual patient scans to population atlas for statistical analysis
- **Label Propagation**: Transfer expert annotations from template to new patients (semi-automated segmentation)
- **Multi-Modal Registration**: Extend approach to align CT, MRI, PET for comprehensive diagnosis
- **Surgical Planning**: Register pre-operative and intra-operative scans for image-guided intervention

## 🔗 References & Further Reading

- **ANTs Manual**: https://github.com/stnava/ANTsDoc/raw/master/ants2.pdf
- **ANTs GitHub**: https://github.com/ANTsX/ANTs
- **SimpleITK Documentation**: https://simpleitk.readthedocs.io/
- **Mutual Information**: Maes et al. "Multimodality Image Registration by Maximization of Mutual Information" (IEEE TMI, 1997)
- **Affine Registration**: Maintz & Viergever "A Survey of Medical Image Registration" (MedIA, 1998)
- **ITK-SNAP Software**: http://www.itksnap.org/

## ⚠️ Troubleshooting Runtime Issues

| Issue | Solution |
|-------|----------|
| Registration diverges (loss increasing) | Try smaller gradient step (0.005); reduce downsampling factors |
| Incorrect point transformation results | Verify image affine matrices match (origin, spacing, direction) |
| Mask has holes after resampling | Confirm nearest-neighbor interpolation used; check transform file exists |
| ITK-SNAP visualization issues | Verify output NIfTI files are valid; check metadata preservation |
| Memory issues with large images | Register lower-resolution versions; use coarser convergence schedules |


