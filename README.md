## Credit Default Prediction – Machine Learning Project

# 1. Project Overview

This project aims to predict credit card default using the UCI Credit Card Default dataset.

The objective is not only to compare predictive performance, but to build a rigorous machine learning pipeline, including:

Structured preprocessing

Handling class imbalance

Comparison of classical and deep learning models

Cross-validation framework

Cost-sensitive decision analysis

Threshold optimization

The project adopts a decision-oriented perspective, minimizing an asymmetric financial cost function rather than maximizing accuracy alone.

# 2. Repository Structure
   
data/raw/
    UCI_Credit_Card.csv

notebook/
    EDA.ipynb

src/
    data_loader.py
    evaluate.py
    train.py
    
# Folder description

data/raw/
Contains the original dataset.

notebook/EDA.ipynb
Exploratory Data Analysis:

Class imbalance analysis

Distribution study

Correlation matrix

Conditional default analysis

src/train.py
Main training script implementing:

Train/test split (stratified)

Stratified 5-fold cross-validation

Logistic Regression

Random Forest

Gradient Boosting

MLP (Neural Network)

Cost-sensitive threshold optimization

src/evaluate.py
Contains evaluation metrics and confusion matrix logic.

src/data_loader.py
Dataset loading and preprocessing utilities.


# 3. Experimental Framework

Stratified train/test split (80/20)

Stratified 5-fold cross-validation

StandardScaler applied where required

Optional SMOTE for imbalance handling

Threshold optimization based on asymmetric cost:

Cost=cFN​×FN+cFP​×FP

Default ratio used:

c_FN = 5
c_FP = 1


# 4. Running the Project

From the root folder:

cd src
python train.py --model gb

Available models:

lr_base
lr_balanced
lr_smote
rf
gb
mlp

Example:

python train.py --model mlp

# 5. Main Findings

Gradient Boosting achieves the best PR-AUC and lowest financial cost.

Deep Learning (MLP) does not outperform tree-based models on structured tabular data.

Threshold optimization significantly improves decision quality under asymmetric cost.

Class rebalancing modifies recall but does not improve intrinsic discrimination.


# 6. Reproducibility

Fixed random_state in all models

Pipeline-based preprocessing

Isolated test set

Controlled environment
