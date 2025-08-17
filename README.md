Credit Card Fraud Detection
Project Overview

This project aims to detect fraudulent credit card transactions using unsupervised anomaly detection methods. Fraud detection is a highly challenging task due to the extremely imbalanced dataset (fraudulent transactions are less than 0.2% of all transactions).

We explore the data with dimensionality reduction techniques (PCA and t-SNE) for visualization and train models including:

Isolation Forest

One-Class SVM

Autoencoders

The goal is to identify fraudulent transactions while minimizing false positives.

Dataset

Dataset: Credit Card Fraud Detection (Kaggle)

Contains transactions made by European cardholders in September 2013.

Features:

V1–V28 = Result of PCA transformation (sensitive details hidden).

Amount = Transaction amount.

Time = Seconds elapsed between this transaction and the first transaction.

Class = Target variable (0 = normal, 1 = fraud).

Project Workflow
1. Data Exploration & Preprocessing

Load and inspect dataset.

Handle class imbalance.

Apply scaling to features where needed.

2. Dimensionality Reduction & Visualization

PCA (Principal Component Analysis): Reduce dimensions while preserving variance.

t-SNE (t-distributed Stochastic Neighbor Embedding): Visualize fraud vs. non-fraud clusters in 2D/3D space.

3. Model Training

We apply unsupervised anomaly detection algorithms since fraud cases are rare and labeling is difficult in practice:

Isolation Forest: Identifies anomalies by random partitioning.

One-Class SVM: Learns the boundary of “normal” transactions and flags outliers.

Autoencoder (Neural Network): Learns to reconstruct normal transactions; high reconstruction error indicates fraud.

4. Model Evaluation

Metrics focus on handling imbalance:

Confusion Matrix

Precision, Recall, F1-Score

ROC-AUC

PR-AUC (Precision-Recall Curve) (especially important for imbalanced data).

How to Run
1. Clone Repository
git clone https://github.com/jaykay73/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install Dependencies
pip install -r requirements.txt

3. Run Jupyter Notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb

Requirements

Python 3.8+

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

TensorFlow / Keras (for Autoencoders)

Key Learnings

Handling imbalanced datasets with anomaly detection.

Using PCA and t-SNE for dimensionality reduction & visualization.

Comparing multiple unsupervised learning approaches.

Interpreting results using metrics designed for rare event detection.

References

Kaggle: Credit Card Fraud Detection Dataset
