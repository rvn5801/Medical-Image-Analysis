# Medical Image Analysis Homework: VoxelMorph Image Registration

A comprehensive implementation of **deformable image registration** using VoxelMorph neural networks. This project addresses the challenge of aligning brain MRI images to enable downstream medical imaging tasks such as voxel-based morphometry, atlas construction, and atlas-based segmentation.

---

## 📋 Project Overview

This homework assignment focuses on **unpaired (inter-subject) image registration**—a fundamental task in medical imaging where we align one brain volume (moving/source image) to another (fixed/target image) using learned deformation fields.

### Key Objectives:
- Implement a VoxelMorph neural network for automatic deformable image registration
- Learn velocity field integration using the **Scaling and Squaring** algorithm
- Build and train a multi-objective loss function combining image similarity, deformation smoothness, and segmentation-based auxiliary losses
- Evaluate registration quality using the Dice Similarity Coefficient (DSC) metric
- Visualize and analyze pre- and post-registration results

### Why This Matters:
Accurate image registration is essential for:
- **Atlas-based segmentation**: Transferring anatomical labels from a template brain to patient scans
- **Voxel-based morphometry (VBM)**: Detecting regional brain volume differences across populations
- **Longitudinal analysis**: Tracking anatomical changes in individual patients over time
- **Multi-subject studies**: Enabling group-level statistical analysis of neuroimaging data

---

## 🎓 What I Learned

### Core Concepts:
1. **Deformable Image Registration**
   - Understanding spatial transformations and warping in 3D
   - Diffeomorphic vs. non-diffeomorphic deformations
   - Stationary Velocity Fields (SVF) and their properties

2. **VoxelMorph Architecture**
   - U-Net encoder-decoder design for learning displacement fields
   - Flow field generation and integration
   - Spatial transformer networks for image warping

3. **Velocity Field Integration**
   - Implementing the **Scaling and Squaring** algorithm
   - Composing small deformation steps to integrate velocity over time
   - Mathematical foundation: $\phi_1 = \int_0^1 v(\phi_t)dt$ approximated via $\phi_1 = \phi_{1/2} \circ \phi_{1/2}$

4. **Multi-Objective Loss Functions**
   - **Image Similarity Loss** (MSE): Measures how well images align
   - **Smoothness Loss** (Bending Energy): Encourages smooth deformations
   - **Segmentation Loss** (Dice): Auxiliary task to improve anatomical alignment
   - Loss weight balancing and trade-offs between competing objectives

5. **Medical Image Analysis Skills**
   - Loading and preprocessing 3D medical images (NIfTI format)
   - One-hot encoding and label handling for segmentation
   - Image blending techniques for visualization
   - Computing segmentation metrics (Dice Similarity Coefficient)

6. **Deep Learning & PyTorch**
   - Building custom neural network modules
   - Automatic mixed precision (AMP) for efficient GPU training
   - Learning rate scheduling with CosineAnnealing
   - Model evaluation and checkpointing strategies

---

## 🛠️ Technologies & Tools Used

### Programming & Frameworks:
- **Python 3.x** – Primary programming language
- **PyTorch** – Deep learning framework for neural network implementation
- **MONAI** (Medical Open Network for AI) – Specialized medical imaging library providing:
  - Pre-built loss functions (`BendingEnergyLoss`, `DiceLoss`)
  - Medical image metrics (`DiceMetric`)
  - Image transformation utilities (`one_hot`, `blend_images`)
  - Data loading and caching (`Dataset`, `DataLoader`, `CacheDataset`)

### Visualization & Analysis:
- **Matplotlib** – Creating publication-quality plots and visualizations
- **NumPy** – Numerical computations and array operations

### Development Tools:
- **Jupyter Notebook** – Interactive development and documentation
- **TensorBoard** – Training monitoring and loss visualization
- **CUDA/cuDNN** – GPU acceleration for training (recommended)
- **Git** – Version control

### Hardware Considerations:
- **GPU recommended**: NVIDIA CUDA-capable GPU (e.g., V100, A100)
- **VRAM requirement**: Minimum 8GB for batch training on 256³ volumes
- **CPU fallback**: PyTorch automatically falls back to CPU if CUDA unavailable

---

## 📊 Data Description

### Dataset: Neurite OASIS (Open Access Series of Imaging Studies)

**Source**: [OASIS Brains Project](https://oasis-brains.org/)

**Dataset Overview**:
- **Total Subjects**: ~400 (394 training + remaining for validation)
- **Modality**: T1-weighted 3D brain MRI
- **Image Format**: NIfTI (.nii.gz) compressed format
- **Image Dimensions**: 256 × 256 × 256 voxels (affinely aligned and normalized)
- **Voxel Resolution**: Isotropic 1mm³

**Data Components**:

| File | Description | Use Case |
|------|-------------|----------|
| `aligned_norm.nii.gz` | Affinely aligned, normalized brain volumes | Training/Validation images |
| `aligned_seg4.nii.gz` | 4-label coarse segmentation (background, CSF, gray matter, white matter) | Training segmentation labels |
| `aligned_seg35.nii.gz` | 35-label fine segmentation (detailed anatomical structures) | Validation and detailed analysis |

**Data Processing Pipeline**:
1. **Loading**: Images loaded with MONAI's `LoadImaged` transform with `ensure_channel_first=True`
2. **Normalization**: Already performed by dataset creators (aligned to Talairach space)
3. **Batching**: 
   - Training: 2 images randomly sampled per batch (split as fixed + moving)
   - Validation: Pre-paired images (consecutive subjects form registration pairs)
4. **Segmentation Encoding**: Labels converted to one-hot encoding for multi-class loss computation

---

## 📚 Notebooks Description

### `hw4_q2.ipynb` – VoxelMorph Image Registration Network

A complete end-to-end implementation covering:

#### Part 1: Environment Setup & Data Loading
- Installing MONAI and required dependencies
- Downloading and extracting the OASIS dataset (~2GB)
- Creating train/validation data lists with proper file paths

#### Part 2: Data Visualization & Exploration
- Loading sample images and segmentation labels
- Visualizing multi-planar slices (coronal, axial, sagittal)
- Creating blended visualizations of images overlaid with anatomical labels
- Understanding data structure and preprocessing steps

#### Part 3: Network Architecture Implementation
- **Velocity Field Integration**: Implementing the `ScaleAndSquare` class
  - Scaling velocity field by factor $1/2^N$
  - Recursively composing deformation fields via `flow = flow + Warp(flow, flow)`
  - Integrating stationary velocity fields to obtain diffeomorphic transformations

- **U-Net Backbone**: Building encoder-decoder architecture
  - Encoding path with max-pooling downsampling
  - Decoding path with upsampling and skip connections
  - Customizable feature depths and levels

- **VoxelMorph Model**: Integrating components
  - Concatenating fixed and moving images
  - U-Net processing
  - Converting output to displacement field via 3D convolution
  - Optional bidirectional registration setup

- **Spatial Transformer**: Warping images using learned deformation fields
  - Grid generation for resampling
  - Bilinear/Trilinear interpolation
  - Boundary handling and normalization

#### Part 4: Loss Function Design
Implementing a multi-objective loss combining:
- **MSE Loss** (weight: `lam_sim`): Penalizes intensity differences
- **Bending Energy Loss** (weight: `lam_smooth`): Enforces smooth deformations
- **Dice Loss** (weight: `lam_dice`): Auxiliary segmentation alignment loss

#### Part 5: Training Pipeline
- Setting up optimizer (Adam) with learning rate scheduling
- Implementing mixed precision training (AMP) for GPU efficiency
- Training loop with validation at specified intervals
- Model checkpointing: Saving best model by validation Dice score
- TensorBoard logging for monitoring

#### Part 6: Evaluation & Visualization
- Computing Dice Similarity Coefficient before/after registration
- Plotting training loss and validation metrics over epochs
- Visualizing registration results on test data
- Comparing fixed, moving, and registered images with segmentation overlays

---

## 🚀 How to Use / Run the Notebooks

### Prerequisites
- **Python 3.8+** installed
- **Jupyter Notebook** or **JupyterLab**
- **GPU** (optional but recommended for training)
- **~30GB free disk space** (for dataset download)

### Step-by-Step Setup

#### 1. Clone or Download the Repository
```bash
cd "path/to/your/homework/directory"
```

#### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Using conda
conda create -n voxelmorph python=3.10
conda activate voxelmorph

# OR using venv
python -m venv voxelmorph_env
source voxelmorph_env/bin/activate  # On Windows: voxelmorph_env\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install monai-weekly[tensorboard]
pip install matplotlib numpy
pip install jupyter
```

**For CPU-only setup** (slower training):
```bash
pip install torch torchvision torchaudio
```

#### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Then open `hw4_q2.ipynb` in your browser.

#### 5. Run the Notebook
- Execute cells sequentially from top to bottom
- **First execution**: Dataset will automatically download (~2-5 minutes depending on internet speed)
- **Training**: Default configuration trains for 3 epochs. Increase `max_epochs` for better results (recommended: 50+)
- **GPU**: Ensure GPU is detected by running the `!nvidia-smi` cell early in the notebook

### Key Configuration Parameters

```python
# Device settings
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
batch_size = 1
learning_rate = 1e-4
max_epochs = 3  # Increase for production training

# Loss weights (critical for tuning)
lam_sim = 1e0      # Image similarity weight
lam_smooth = 1e-2  # Smoothness regularization weight
lam_dice = 2e-2    # Segmentation auxiliary loss weight

# Model architecture
int_steps = 7      # Number of integration steps (higher = more accurate)
int_downsize = 2   # Downsampling factor for integration
```

### Expected Output

After training completes, you'll have:
- **Trained model weights**: Saved in `models/voxelmorph/` directory
- **TensorBoard logs**: Viewable with `tensorboard --logdir=models/voxelmorph/`
- **Visualizations**: Plots of training loss and validation metrics
- **Registration results**: Pre/post-registration image comparisons

### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` or `int_steps`, or use `int_downsize > 2` |
| Slow data loading | Enable multi-worker loading with `num_workers=4` in DataLoader |
| Model not improving | Increase `max_epochs`, adjust loss weights, or use pretrained weights |
| Dataset won't download | Check internet connection or manually download from [OASIS website](https://oasis-brains.org/) |

---

## 📈 Results & Validation Metrics

The model is evaluated using:
- **Dice Similarity Coefficient (DSC)**: Measures overlap between moving label and fixed label after registration
  - Range: 0 to 1 (1 = perfect alignment)
  - Computed before and after registration to quantify improvement
  - Higher improvement indicates better registration quality

---

## 📖 References

1. **Original VoxelMorph Paper**: [An Unsupervised Learning Model for Deformable Medical Image Registration](https://arxiv.org/pdf/1809.05231.pdf)

2. **OASIS Dataset**: Lamontagne et al. (2019). "Open Access Series of Imaging Studies (OASIS)". https://oasis-brains.org/

3. **MONAI Documentation**: [Medical Open Network for AI](https://docs.monai.io/)

4. **Scaling and Squaring for Diffeomorphic Registration**: Arsigny et al. (2006)

---

## 💡 Future Improvements

- Implement bidirectional registration for symmetric alignment
- Add multiple registration metrics (Hausdorff distance, log Jacobian smoothness)
- Experiment with different U-Net architectures and loss weight combinations
- Extend to multi-resolution coarse-to-fine registration
- Deploy model as a web service for clinical applications

---




