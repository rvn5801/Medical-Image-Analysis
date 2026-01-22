# Medical Image Analysis: Homework 1

## Project Overview

This repository contains homework assignments focused on **machine learning fundamentals** applied to medical imaging data. The primary objectives are to:

- Develop a deep understanding of **linear regression** and **logistic regression** from mathematical foundations to implementation
- Apply machine learning techniques to **real-world medical imaging data** (neonatal brain imaging)
- Implement algorithms from scratch using vectorized operations, gaining insights into how popular ML frameworks work under the hood
- Analyze the relationship between **brain volume measurements** and **birth outcomes** (preterm vs. term delivery)

The assignments progressively build from theoretical matrix mathematics to practical classification tasks, emphasizing both mathematical rigor and computational efficiency.

---

## What I Learned

### Core Concepts & Skills

**1. Linear Regression Fundamentals**
- Formulating regression problems in matrix notation
- Understanding the Normal Equation: $(X^T X)w = X^T y$
- Recognizing rank deficiency and non-invertible matrices in real-world data
- Identifying when linear systems have no unique solution (e.g., collinear features)

**2. Logistic Regression & Classification**
- Forward propagation in neural networks (linear transformation → sigmoid activation)
- Sigmoid function properties and its role in probability estimation
- Cross-entropy loss function for classification tasks
- Gradient descent optimization and backpropagation fundamentals
- Model evaluation using accuracy metrics and decision thresholds

**3. Medical Image Analysis**
- Understanding cortical brain imaging metrics: **cortical thickness**, **cortical folding**, and **cortical myelination**
- Working with **brain region-of-interest (ROI)** data on cortical surfaces
- Preprocessing and normalization of high-dimensional medical data
- Class imbalance handling (900 term vs. 100 preterm neonates)

**4. Programming & Data Science Skills**
- Vectorized NumPy operations for efficient computation
- Data visualization with Matplotlib (histograms, decision boundaries)
- Train-test split and cross-validation principles
- Working with pickle-serialized datasets
- Mathematical notation and scientific computing

---

## Technologies & Tools Used

### Programming Languages & Environments
- **Python 3.x** – Core programming language
- **Jupyter Notebook (.ipynb)** – Interactive development and documentation
- **Google Colab** – Cloud-based execution environment with pre-installed libraries

### Libraries & Frameworks
- **NumPy** – Numerical computing, matrix operations, and vectorization
- **Pandas** – Data manipulation and analysis (optional for analysis)
- **Matplotlib** – Data visualization and plotting
- **scikit-learn** – Train-test splitting and model evaluation utilities
- **Pickle** – Data serialization for loading datasets

### Key Tools
- Mathematical notation and LaTeX for formula representation
- Google Drive integration for data access in Colab
- Git & GitHub for version control and collaboration

---

## Data Description

### Dataset Overview
**Source:** Neonatal brain imaging study comparing term-equivalent and preterm infants  
**Format:** Pickle-serialized Python dictionary  
**File:** `data_hw1_q1.pkl`

### Data Characteristics
| Aspect | Details |
|--------|---------|
| **Total Samples** | 1,000 neonates |
| **Class Distribution** | 900 term births, 100 preterm births |
| **Features per Domain** | 100 regions of interest (ROIs) × 3 imaging metrics = 300 features |
| **Feature Types** | Continuous brain volume measurements (cortical thickness, folding, myelination) |
| **Target Variable** | Binary classification: Term (y=0) vs. Preterm (y=1) |
| **Data Type** | Floating-point measurements from brain surface imaging |

### Data Processing
1. **Loading:** Pickle deserialization to extract feature matrix (X) and labels (y)
2. **Transposition:** Data transposed from (samples × features) to (features × samples) format for linear algebra operations
3. **Bias Term:** Added row of ones to feature matrix to account for intercept in regression
4. **Train-Test Split:** 90% training (900 samples), 10% testing (100 samples) using stratified random split
5. **No explicit scaling:** Features analyzed in original units; each feature represents averaged brain volume measurements

### Data Challenge
The study exhibits **class imbalance** (90:10 ratio), requiring careful evaluation of both accuracy and sensitivity/specificity metrics.

---

## Notebooks Description

### **Homework 1 (116066299_hw1.ipynb)**

A comprehensive assignment covering two fundamental machine learning algorithms:

#### **Question 1: Linear Regression (30 points)**
- **Problem Setup:** Solve a linear regression task with 100 samples and 4 features
- **Key Tasks:**
  - Formulate regression loss in matrix form: $\mathcal{L}(\mathbf{w}) = (y - Xw)^T(y - Xw)$
  - Derive the Normal Equation by taking derivatives with respect to weights
  - **Critical Analysis:** Determine invertibility of the system matrix $A = X^T X$
  - Recognize rank deficiency: probability outputs from a 4-class classifier are inherently linearly dependent
  - Bonus: Calculate the smallest eigenvalue (zero for singular matrices)
- **Concepts Reinforced:** Matrix calculus, rank deficiency, non-invertible systems, eigenvalue analysis

#### **Question 2: Logistic Regression (40 points)**
- **Real-World Dataset:** 1,000 neonates with 300 brain imaging features predicting preterm birth
- **Major Tasks:**

  **2.1 – Vectorized Linear Transformation**
  - Implement $Z = WX$ for efficient batch processing
  - Understand matrix dimensions: weights $(1 × m)$ × features $(m × n)$ = predictions $(1 × n)$

  **2.2 – Sigmoid Activation**
  - Implement sigmoid function: $f(z) = \frac{1}{1 + e^{-z}}$
  - Verify function behavior (S-shaped curve, range [0,1])
  - Understand probabilistic interpretation for classification

  **2.3 – Cross-Entropy Loss**
  - Define binary cross-entropy loss for optimization
  - Reason about loss landscapes and gradient descent convergence

  **2.4 – Backpropagation** (implementation details in notebook)
  - Compute gradients with respect to weights using chain rule
  - Update weights iteratively toward optimal solution

  **2.5 – Model Training & Evaluation**
  - Train logistic regression classifier using gradient descent
  - Monitor loss and accuracy during training
  - Evaluate performance on independent test set
  - Analyze decision boundaries and threshold selection

- **Skills Developed:** Forward/backward propagation, gradient computation, iterative optimization, model evaluation

---

## How to Use / Run the Notebooks

### Prerequisites
- **Python 3.6+** installed locally, OR
- **Google Colab** account (recommended for simplicity)

### Option 1: Google Colab (Recommended)
Google Colab has most dependencies pre-installed. This is the **intended environment** for these assignments.

1. **Download the notebook** (`116066299_hw1.ipynb`) to your computer
2. **Go to [Google Colab](https://colab.research.google.com/)**
3. **Upload the notebook:** File → Open notebook → Upload tab → Select `.ipynb` file
4. **Add your Google Drive data:**
   - Place `data_hw1_q1.pkl` in your Google Drive under `/MyDrive/Medical Image Analysis/`
   - Or modify the data path in the notebook to point to your data location
5. **Run cells sequentially:** Click the play button on each cell (or Ctrl+Enter)

### Option 2: Local Environment (Python + Jupyter)

#### Installation
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install jupyter numpy matplotlib scikit-learn
```

#### Running the Notebook
```bash
# Navigate to the notebook directory
cd "path/to/homework/directory"

# Start Jupyter Notebook
jupyter notebook

# Open 116066299_hw1.ipynb in your browser and run cells
```

#### Data Setup (Local)
- Download `data_hw1_q1.pkl` from the course materials
- Place it in the same directory as the notebook OR
- Update the data loading path in the code:
  ```python
  data = pickle.load(open('path/to/data_hw1_q1.pkl', 'rb'))
  ```

### Running Individual Cells
- Each cell is self-contained and depends on previously executed cells
- Execute cells in order (top to bottom) to maintain variable state
- Use **Shift+Enter** to run a cell and move to the next one
- Use **Ctrl+Enter** (Cmd+Enter on Mac) to run without advancing

### Expected Runtime
- Total execution time: **5-15 minutes** depending on system
- Bottlenecks: Data loading and gradient descent iterations (Question 2)

---

## Key Takeaways for Recruiters

This assignment demonstrates:

✅ **Mathematical Foundations** – Strong grasp of linear algebra, calculus, and optimization  
✅ **Algorithm Implementation** – Building ML models from scratch (not just using libraries)  
✅ **Real-World Application** – Applying theory to authentic medical imaging data  
✅ **Code Quality** – Vectorized, efficient NumPy code; clear documentation  
✅ **Problem-Solving** – Recognizing mathematical constraints (rank deficiency) and their implications  
✅ **Scientific Communication** – Clear mathematical notation and result interpretation  

---

