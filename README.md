# Credit Risk Analysis: Robustness & Data Drift Mitigation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TON_LIEN_COLAB_ICI)

## 📌 Project Overview
This project implements a complete end-to-end Machine Learning pipeline to predict loan default risks. Beyond standard classification, the core focus is on **model reliability** and **lifecycle management**. It simulates real-world scenarios where data distributions evolve over time (Data Drift) and evaluates multiple strategies to maintain model accuracy.

## 🛠 Technical Workflow (Based on Notebook)
The project follows a structured 8-phase methodology:
1. **Data Preprocessing:** Handling outliers (e.g., person_age, person_emp_length) and encoding categorical variables.
2. **Baseline Modeling:** Training an optimized **XGBoost** classifier.
3. **Robustness Evaluation:** Using **Bootstrap** sampling to calculate 95% Confidence Intervals for ROC-AUC.
4. **Interpretability:** Utilizing **SHAP values** to explain global model behavior and individual credit decisions.
5. **Drift Simulation:** Artificial introduction of data drift (e.g., shifting income or age distributions).
6. **Drift Detection:** Implementing statistical tests (**Kolmogorov-Smirnov**) to identify feature distribution shifts.
7. **Mitigation Strategies:** - **Strategy A:** Dropping drifted features.
    - **Strategy B:** Model retraining on new data domains.
8. **Final Comparison:** Comprehensive benchmarking of performance across all scenarios.

## 📊 Key Results
As shown in the final analysis, the pipeline effectively identifies performance degradation due to drift (dropping from ~0.87 to ~0.80 AUC) and successfully recovers performance (~0.88 AUC) through automated retraining.

## 📂 Project Structure
* **SaveProjet.ipynb**: The main research and execution notebook.
* **src/**: Modular Python scripts for preprocessing and drift detection.
* **requirements.txt**: List of dependencies (xgboost, shap, scikit-learn, etc.).
* **README.md**: Project documentation.

## 🚀 Usage
1. **Quick Run:** Click the "Open in Colab" badge to see the full execution and final comparative table.
2. **Requirements:**
   ```bash
   pip install xgboost shap scikit-learn matplotlib seaborn
