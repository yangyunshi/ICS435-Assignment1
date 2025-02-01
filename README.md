# ICS435-Assignment1
# Breast Cancer Classification with KNN, Decision Tree, and Random Forest

## Overview

This project explores the application of three different classification models - K-Nearest Neighbors (KNN), Decision Tree, and Random Forest - to predict the diagnosis of breast cancer using the scikit-learn Breast Cancer dataset.

## Data Preprocessing

The script performs the following preprocessing steps:

1. **Loading Data:** Loads the Breast Cancer dataset using `load_breast_cancer` from `sklearn.datasets`.
2. **Data Splitting:** Partitions the data into an 80% training set and a 20% test set using `train_test_split` from `sklearn.model_selection`.
3. **Feature Scaling:** Scales the features using `StandardScaler` from `sklearn.preprocessing` to ensure features with different ranges do not disproportionately influence the models.

## Model Training

The script trains three classifiers:

1. **K-Nearest Neighbors (KNN):** Starts with `n_neighbors=5`.
2. **Decision Tree:** Uses the default settings initially and then experiments with `max_depth`.
3. **Random Forest:** Starts with 100 trees (`n_estimators=100`) and explores the effect of different `max_depth` or `min_samples_split`.

## Evaluation

The script evaluates the models using the following metrics:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-score**

It also includes a confusion matrix for each model and compares the results across the models using a bar graph.

## Ablation Study

The script performs an ablation study by modifying key hyperparameters (e.g., `n_neighbors` for KNN, `max_depth` for Decision Trees and Random Forest) and observes their impact on performance. This is done to identify the optimal hyperparameter values for each model.

## Requirements

* Python 3.x
* scikit-learn
* NumPy
* Matplotlib

## Usage

1. Make sure you have the required libraries installed. You can install them using pip:
   ```bash pip install scikit-learn numpy matplotlib```
3. Run the Python script in a Google Colab environment or Jupyter Notebook.
4. The script will train the models, evaluate their performance, and display the results.

## Author
Allison Ebsen
