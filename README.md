# Cuckoo Algorithm for Corporate Financial Risk Prediction (FRPF-CSO)

This repository contains the full implementation and replication of the research paper:

**"Using Cuckoo Search Algorithm to Predict Corporate Financial Risks and Alleviate Economic Uncertainty"**  
International Journal of Computational Intelligence Systems, Vol. 18, 2025  
Author: Muqiao Cai  

---

## Project Overview

This project aims to replicate and extend the Financial Risk Prediction Framework utilizing Cuckoo Search Optimization (FRPF-CSO).  
The original paper proposes a hybrid model that combines a Backpropagation Neural Network (BPNN) with the Cuckoo Search Algorithm (CSA) to predict corporate financial risk under high economic uncertainty.

The implementation follows the same structure and metrics as the original study, adding interpretability tools and an adjustable cost-based threshold for business-oriented decision-making.

---

## Architecture

### 1. Data Pre-Processing
- Loads and cleans financial data (.dta / .csv).  
- Handles missing values via median imputation and IQR-based outlier filtering.  
- Applies Min–Max normalization and SMOTE to address class imbalance.

### 2. Model Construction
- BPNN architecture with one hidden layer, ReLU activation, and dropout regularization.  
- CSA for global optimization of:
  - Hidden layer size  
  - Learning rate  
  - Dropout rate  
  - Epochs  
  - Feature subset  

### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, AUC  
- Risk Detection Ratio (RDR), RMSE, MAE, Convergence Rate  
- Custom cost-sensitive thresholding for different business scenarios

---

## Repository Structure

Cuckoo_Algorithm_Financial_Risk/
│
├── data/                    # Raw and processed datasets
├── scripts_notebooks/       # Main Jupyter notebooks and source code
├── artifacts/               # Stored models, metrics, and generated plots
├── agent_workspace/         # Auxiliary workspace (if used)
├── requirements.txt         # Dependencies for environment setup

---

## Installation

To replicate the experiment, create a virtual environment and install dependencies:

pip install -r requirements.txt


If you use Conda:

conda create -n frpf_cso python=3.11
conda activate frpf_cso
pip install -r requirements.txt