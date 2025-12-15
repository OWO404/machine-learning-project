# machine-learning-project
## Objective
This project aims to solve a regression or classification problem using a rigorous machine learning pipeline.
Both classical machine learning models and deep learning approaches are implemented and compared.
A custom component (metric or loss) is also included.
## Project Structure
data/
│── raw/ # Raw datasets
│── processed/ # Processed datasets

notebooks/
│── 01_eda.ipynb # Exploratory Data Analysis (EDA only)

src/
│── data_loader.py
│── preprocessing.py
│── models/
│ │── classical.py
│ │── neural_net.py
│── train.py
│── evaluate.py

## Methodology
- Data exploration and preprocessing
- Classical ML models (baseline and tree-based)
- Deep learning model (MLP)
- Cross-validation and hyperparameter tuning
- Model comparison and critical analysis

## Tools
- Python
- Scikit-learn
- PyTorch / TensorFlow
