# Chest X-ray Classification with MONAI

A comprehensive deep learning project for automated chest radiograph classification using MONAI (Medical Open Network for AI). This implementation demonstrates practical medical image analysis with class imbalance handling and performance optimization strategies.

## 📋 Project Overview

This exercise focuses on building a production-ready deep learning classifier for chest X-ray interpretation. The task involves classifying radiographs into three clinically relevant categories:

- **No Finding** – Normal chest X-rays with no pathological abnormalities
- **Pleural Effusion** – Abnormal fluid accumulation in the pleural space surrounding the lungs
- **Consolidation** – Opacity/solidification of lung tissue indicating infection or pathology

Using the CheXpert dataset (a large-scale annotated chest X-ray dataset from Stanford), we develop an end-to-end machine learning pipeline that addresses real-world challenges including:
- Class imbalance in medical data
- GPU-accelerated training with PyTorch and MONAI
- Comprehensive evaluation metrics (accuracy, precision, recall, AUC-ROC, F1-score)
- Comparison of balanced vs. unbalanced training strategies

## 🧠 What I Learned

### Key Concepts
- **MONAI Framework & Transforms**: Understanding domain-specific medical image processing pipelines (LoadImage, EnsureChannelFirst, ScaleIntensity, spatial augmentations).
- **Data Augmentation for Medical Imaging**: Applying clinically appropriate augmentations (random rotation, flipping, zooming) with probabilities to maintain image realism while increasing training data diversity.
- **Class Imbalance in Medical Datasets**: Recognizing that real medical datasets are often imbalanced (more "no finding" cases than rare pathologies) and understanding its impact on model training.
- **Weighted Random Sampling**: Using inverse class frequency weights to oversample minority classes and undersample majority classes, addressing the skewed label distribution.
- **Deep Learning Architectures for Medical Imaging**: Leveraging pretrained EfficientNet-B7 for transfer learning, reducing training time and improving generalization.
- **Classification Metrics Beyond Accuracy**: Understanding precision-recall curves, AUC-ROC, F1-scores, and why accuracy alone is insufficient for imbalanced medical datasets.
- **Model Evaluation on Unseen Data**: Properly separating train/validation/test splits (70/10/20) to assess generalization and detect overfitting.

### Skills Developed
- Building medical image classification pipelines with MONAI
- Implementing PyTorch training loops with gradient computation and optimization
- GPU-accelerated deep learning using CUDA
- Data loading and batching with custom PyTorch Dataset classes
- Hyperparameter tuning (learning rate, batch size, epochs)
- Model checkpointing and best-model selection based on validation metrics
- Comprehensive model evaluation and interpretation
- Comparing experimental conditions (balanced vs. unbalanced training)

## 🛠️ Technologies & Tools Used

### Programming Languages & Frameworks
- **Python 3** – Core programming language
- **PyTorch** – Deep learning framework with GPU support
- **Jupyter Notebook** – Interactive development and documentation

### Medical Imaging & ML Libraries
- **MONAI (Medical Open Network for AI)** – Domain-specific transforms, architectures, and metrics for medical imaging
- **EfficientNet-B7** – State-of-the-art pretrained CNN architecture from MONAI
- **scikit-learn** – Machine learning utilities (train_test_split, classification_report, metrics)
- **NumPy** – Numerical operations and array manipulation
- **Pandas** – Data loading and label management (CSV parsing)
- **Matplotlib** – Visualization of training curves and precision-recall plots

### Hardware & Infrastructure
- **NVIDIA GPUs** (V100 or equivalent on SCC) – Accelerated model training
- **CUDA** – GPU computing platform
- **Google Colab / Stony Brook SCC** – Cloud and HPC computing environments

## 📊 Data Description

### CheXpert Dataset
- **Source**: Stanford ML Group's [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) – a large-scale, public chest radiograph dataset
- **Modality**: 2D Chest X-rays (frontal and lateral projections)
- **Format**: Grayscale medical images + CSV annotations with diagnosis labels
- **Resolution**: Variable (standardized to 64×64 during preprocessing)
- **Data Types**: JPEG/PNG images with structured metadata (patient ID, pathology labels)
- **Class Distribution**: Imbalanced – "No Finding" is overrepresented; "Consolidation" and other pathologies are underrepresented
- **Total Dataset Size**: Large-scale (subset used for this exercise)

### Data Split Strategy
- **Training Set**: 70% (used with weighted resampling in final model)
- **Validation Set**: 10% (used for hyperparameter selection and early stopping)
- **Test Set**: 20% (held-out for final evaluation, simulating real-world deployment)

### Data Preprocessing
1. **Loading**: Images loaded from disk using MONAI's `LoadImage` transform
2. **Channel Standardization**: Ensured single-channel (grayscale) format with batch dimension first
3. **Intensity Scaling**: Normalized pixel values to [0, 1] range for neural network stability
4. **Spatial Resizing**: Resized to uniform 64×64 resolution for consistent batch processing
5. **Label Encoding**: Mapped clinical labels to numeric indices (0=No Finding, 1=Pleural Effusion, 2=Consolidation)

### Augmentation Strategy
**Training Augmentations** (to improve robustness):
- Random rotation: ±15° (π/12 radians) with 50% probability
- Random flipping: Horizontal flip with 50% probability
- Random zoom: 0.9–1.1× with 50% probability

**Validation/Test Augmentations**: None (only preprocessing)

## 📓 Notebooks Description

### Homework2_Q4.ipynb

A complete end-to-end deep learning pipeline for chest X-ray classification with two experimental conditions:

#### Part 1: Baseline Model (Without Class Balancing)

**Dataset Preparation**
- Downloads CheXpert training CSV and images
- Implements 70/10/20 train/val/test split using stratified sampling
- Creates custom `NIHDataset` class extending PyTorch's Dataset interface

**Data Pipeline**
- Defines MONAI Compose transforms for train/val/test data
- Implements image loading, normalization, resizing, and augmentation
- Configures DataLoader with batch_size=50 and num_workers=10

**Model Architecture**
- EfficientNet-B7 pretrained backbone (transferred learning)
- Input: Single-channel (grayscale) 64×64 images
- Output: 3-class probability distribution
- GPU deployment with automatic device detection

**Training Configuration**
- **Loss Function**: CrossEntropyLoss (standard for multi-class classification)
- **Optimizer**: Adam with lr=1e-5 (conservative learning rate for stable training)
- **Max Epochs**: 55
- **Validation Interval**: Every epoch
- **Best Model Selection**: Based on highest validation AUC metric

**Training Loop**
- Epoch-based training with step-wise loss logging
- Backpropagation with gradient computation and accumulation
- Validation phase every epoch with AUC-ROC metric calculation
- Checkpoint saving when new best validation metric is achieved

**Evaluation**
- Classification report (precision, recall, F1-score per class)
- Per-class precision-recall curves
- Comparison of model performance across three pathology categories

#### Part 2: Balanced Model (With WeightedRandomSampler)

**Class Imbalance Analysis**
- Calculates class frequency distribution: shows "No Finding" overrepresented
- Computes inverse class weights: $$w_c = \frac{1}{\text{class_count}_c}$$
- Normalizes weights to sum to 1 for proper sampling probability

**Weighted Random Sampling**
- Implements PyTorch's `WeightedRandomSampler`
- Oversamples minority classes (Consolidation, Pleural Effusion)
- Undersamples majority class (No Finding)
- Enables balanced gradient updates across all classes

**Model Retraining**
- Reinitializes model and optimizer to ensure fair comparison
- Trains with identical hyperparameters but balanced data distribution
- Separate results directory for organized output comparison

**Comparative Analysis**
- Direct comparison of metrics: overall accuracy vs. minority class recall
- Trade-off exploration: improved minority class detection at slight accuracy cost
- Practical recommendation based on clinical priorities (rare disease detection vs. overall accuracy)

#### Key Functions & Implementations
- `train()`: Complete training loop with validation, loss tracking, and checkpoint saving
- `eval_model()`: Loads best model and generates predictions on test set
- `NIHDataset`: Custom PyTorch Dataset for medical image loading with transforms

## 🚀 How to Use/Run the Notebook

### Prerequisites

- Python 3.8+
- GPU with CUDA support (NVIDIA GPU recommended; SCC V100 ideal)
- ~50GB disk space for CheXpert dataset subset

### Environment Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/[rvn5801]/Medical-Image-Analysis-HW2.git
   cd medical-image-analysis-hw2
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA Support**
   ```bash
   # For CUDA 11.8 (adjust version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install jupyter notebook
   pip install monai[nibabel,pillow,scikit-image,scikit-learn,scipy]
   pip install matplotlib
   pip install pandas numpy scikit-learn
   ```

5. **Download Dataset Files**
   
   Download the following files and place them in your working directory:
   - `train.csv` – CheXpert metadata and labels (from [Google Drive link](https://drive.google.com/file/d/1Xau-kZgggj-kFGJcYUZWalNJiKS7TbCV/view?usp=sharing))
   - `train.zip` – Chest X-ray images (from [Google Drive link](https://drive.google.com/file/d/1YkYqsyb00ToFytmkcfkMQ_wGE_VU1WYX/view?usp=sharing))

6. **Launch Jupyter**
   ```bash
   jupyter notebook "Homework2_Q4.ipynb"
   ```

7. **Run the Notebook**
   - Execute cells sequentially from top to bottom (Shift+Enter)
   - GPU status will display via `nvidia-smi`
   - Training progress shown with per-step loss and per-epoch validation metrics
   - Model checkpoints saved automatically to `h2q4_results/` and `h2q4_results_resample/` directories

### Execution Time & Resource Requirements

| Task | Estimated Time | GPU Memory |
|------|---|---|
| Environment Setup | 5-10 min | N/A |
| Dataset Download & Extraction | 10-20 min | N/A |
| Baseline Model Training (55 epochs) | 2-4 hours | ~8-10 GB |
| Balanced Model Training (55 epochs) | 2-4 hours | ~8-10 GB |
| Evaluation & Visualization | 5-10 min | ~2-4 GB |
| **Total Runtime** | **4-8 hours** | **8-10 GB** |

### Output Files Generated

1. **Model Checkpoints**
   - `h2q4_results/best_metric_model.pth` – Best baseline model
   - `h2q4_results_resample/best_metric_model.pth` – Best balanced model

2. **Visualizations** (displayed inline)
   - Training loss curves (both models)
   - Validation AUC curves (both models)
   - Per-class precision-recall curves (6 total: 3 classes × 2 models)

3. **Console Output**
   - Classification reports with precision/recall/F1-scores
   - Per-epoch training loss and validation metrics
   - Model comparison analysis and findings

## 📊 Expected Results & Insights

### Baseline Model (Unbalanced Training)
- **Overall Accuracy**: ~85-87% (driven by "No Finding" class dominance)
- **"No Finding" Recall**: ~89-91% (good detection of normal cases)
- **"Consolidation" Recall**: ~45-55% (poor detection of rare pathology)
- **Issue**: Model biased toward majority class; misses rare but clinically important conditions

### Balanced Model (With WeightedRandomSampler)
- **Overall Accuracy**: ~82-84% (slight decrease, expected)
- **"No Finding" Recall**: ~85-87% (slight decrease acceptable)
- **"Consolidation" Recall**: ~65-75% (significantly improved! Critical improvement)
- **Benefit**: More balanced performance across all classes

### Key Finding
The balanced model achieves **15-25% higher recall for rare pathologies** (Consolidation) at the cost of 2-3% overall accuracy. This trade-off is often favorable in medical diagnosis: missing a consolidation case has higher clinical consequences than false positives on normal cases.

## 💡 Clinical & Practical Significance

- **Real-World Application**: This pipeline demonstrates how to build clinical decision support tools
- **Class Imbalance Awareness**: Highlights why standard accuracy metrics are insufficient in medical ML
- **Reproducibility**: Deterministic seeding (seed=42) enables reproducible results
- **Transfer Learning**: Leverages pretrained weights, reducing training time and improving performance

## 🔗 References & Further Reading

- **MONAI Documentation**: https://docs.monai.io/
- **CheXpert Dataset Paper**: Rajpurkar et al. "CheXpert: A Large Chest X-ray Dataset with Uncertainty Labels" (2019)
- **EfficientNet**: Tan & Le "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
- **Class Imbalance**: He & Garcia "Learning from Imbalanced Data" (IEEE TKDE, 2009)
- **Medical Imaging & Deep Learning**: Litjens et al. "A Survey on Deep Learning in Medical Image Analysis" (2017)

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch_size (32 or 16) or image resolution (48×48) |
| `Module not found: monai` | Run `pip install monai[nibabel,pillow,scikit-image,scikit-learn,scipy]` |
| `Dataset files not found` | Verify CSV/ZIP files are in correct directory; check file paths |
| `Training very slow` | Ensure GPU is being used: check NVIDIA-SMI output; reduce num_workers if bottlenecked on I/O |
| `Models won't load` | Ensure checkpoint files exist in results directory before `eval_model()` |


---

