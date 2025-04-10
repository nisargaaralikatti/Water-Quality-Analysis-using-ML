# 💧 Water Quality Prediction using ML

<p align="center">
<img src="https://c.tenor.com/sukdxuAeg_QAAAAC/water.gif" alt="Water GIF" width="700" height="350">
</p>

<h1 align="center">💦 Water Potability Prediction 💦</h1>

<div align="center">
    <a href="https://colab.research.google.com/drive/1R2n2CplVKxBFr6I7NlGH_Zd0O7ES8AP5?usp=sharing">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    </a>
    <img src="https://img.shields.io/badge/Language-Python-blue?style=flat&logo=python&logoColor=white" alt="Python Badge">
</div>

## 🧭 Contents
- [Overview](#📌-overview)
- [Project Goal](#🎯-project-goal)
- [Workflow](#🔁-workflow)
- [Data Source](#🧪-data-source)
- [Handling Imbalanced Data](#🧬-handling-imbalanced-data)
- [Tools & Libraries](#🛠️-tools--libraries)
- [ML Models Applied](#📚-ml-models-applied)


## 📌 Overview
Access to clean water is essential for human well-being. Polluted water sources can carry a variety of harmful pathogens and chemicals, leading to health issues like cholera, typhoid, and hepatitis. Identifying whether a water sample is safe (potable) or not is critical, and this project leverages machine learning to do exactly that.


## 🎯 Project Goal
To utilize machine learning algorithms for identifying whether a given sample of water is drinkable or not based on a range of chemical and physical parameters.

## 🔁 Workflow

<p align="center">
<img src="https://user-images.githubusercontent.com/63184114/137891107-ebe26789-bd93-4724-8192-45a7d82dea85.png" width="500" height="350">
</p>

## 🧪 Data Source
The dataset used in this project is available [here on Kaggle](https://www.kaggle.com/adityakadiwal/water-potability). It contains measurements from 3,276 water samples.

**Features include:**
- `ph`: Acidity or alkalinity level
- `Hardness`: Mineral content
- `Solids`: Total dissolved solids
- `Chloramines`: Disinfectant presence
- `Sulfate`: Concentration in mg/L
- `Conductivity`: Ion conductivity
- `Organic_carbon`: Organic carbon concentration
- `Trihalomethanes`: Disinfection by-products
- `Turbidity`: Clarity of the water
- `Potability`: Target variable (1 = safe, 0 = unsafe)

python
import pandas as pd
df = pd.read_csv('water_potability.csv')


## 🧬 Handling Imbalanced Data
The dataset exhibited a class imbalance, making it challenging for models to learn effectively. To address this, **SMOTE** (Synthetic Minority Oversampling Technique) was applied. It artificially generates new instances of the minority class by interpolating between existing ones.

## 📚 ML Models Applied

### 🔹 Logistic Regression
A statistical model using the sigmoid function to classify whether water is potable or not.

<p align="center">
<img src="https://miro.medium.com/max/640/1*OUOB_YF41M-O4GgZH_F2rw.png" width="500" height="350">
</p>


### 🔹 Support Vector Classifier (SVC)
This model seeks to identify a hyperplane that distinctly classifies the data points into potable and non-potable.

<p align="center">
<img src="http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1526288453/index3_souoaz.png" width="500" height="350">
</p>

### 🔹 Random Forest Classifier
A powerful ensemble method combining the predictions of multiple decision trees to improve accuracy and reduce overfitting.

<p align="center">
<img src="https://www.freecodecamp.org/news/content/images/2020/08/how-random-forest-classifier-work.PNG" width="500" height="350">
</p>

### 🔹 XGBoost
A high-performance boosting algorithm that efficiently trains gradient-boosted decision trees.

## 🛠️ Tools & Libraries

The project is built using the following Python libraries:

python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.offline as pyo
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt

- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Visualization libraries
- **Plotly**: Interactive visualizations
- **Scikit-learn**: ML modeling and evaluation
- **PyMC3 & Theano**: For probabilistic modeling and backend tensor operations
