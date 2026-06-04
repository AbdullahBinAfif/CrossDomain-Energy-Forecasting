# Cross-Domain Energy Forecasting
**A Transfer Learning Approach for Data-Scarce Building Environments**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official repository for the Master's dissertation:** *Cross-Domain Energy Forecasting*. 
> This project implements an XGBoost Transfer Learning pipeline to mitigate domain shift and data scarcity in building energy prediction.

---

## 1. Project Abstract
The building sector accounts for a substantial proportion of global energy consumption. Accurate, high-resolution forecasting is a foundational requirement for smart grid optimization. However, traditional predictive frameworks exhibit a pronounced vulnerability to **domain shift**; they require extensive historical datasets to maintain accuracy. When deployed in "data-scarce" environments (e.g., newly commissioned facilities or buildings transitioning to 24/7 continuous operation), these models frequently fail to generalize.

This research investigates the application of **Transfer Learning (TL)** to mitigate these limitations. Utilizing an Extreme Gradient Boosting (XGBoost) architecture, this project empirically evaluates the transfer of foundational thermodynamic and temporal feature weights from a highly structured, data-abundant source domain (**Educational facilities**) to a complex, continuous-operation target domain (**Healthcare facilities**), successfully reducing predictive error using only a fractional subset of target data.

---

## 2. Dataset Access (ASHRAE Great Energy Predictor III)
Due to GitHub's file size limitations, the ~2.6 GB raw dataset is **not** hosted in this repository. To reproduce this experiment, you must download the dataset directly from Kaggle.

1. Navigate to the [ASHRAE - Great Energy Predictor III Kaggle Competition](https://www.kaggle.com/c/ashrae-energy-prediction/data).
2. Download the following files:
   - `train.csv`
   - `building_metadata.csv`
   - `weather_train.csv`
3. Place these files in a local directory named `ashrae_data/` at the root of this project.

---

## 3. Reproducibility Setup
This experiment was engineered to run efficiently in **Google Colab** (utilizing a T4 GPU), but can be executed locally. 


Project Structure
CrossDomain-Energy-Forecasting/
│
├── ashrae_data/                  # (Locally hosted data directory)
├── models/                       # Exported .json model weights
│   └── teacher_model_repro.json  # Pre-trained base model
│
├── notebooks/
│   ├── 01_Data_Engineering.ipynb # Memory optimization, imputation, feature extraction
│   ├── 02_Baseline_Model.ipynb   # Teacher model training on Source Domain
│   └── 03_Transfer_Learning.ipynb# Target adaptation, sweeps (1%-20%), and final evaluation
│
├── visuals/                      # Output graphs and EDA charts
└── README.md
**Required Dependencies:**
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
