# Crop-Yield-Prediction-Module1
# Crop Yield Prediction - Module 1 (Preprocessing & EDA)

This module handles the Exploratory Data Analysis (EDA) and data preprocessing pipeline for the Crop Yield Prediction project. It cleans the raw data, visualizes feature distributions, and exports a trained preprocessor object.

## 📁 Repository Structure
* `modul1_preprocessing.py`: Main Python script containing data cleaning and preprocessing logic.
* `preprocessor.joblib`: Serialized binary file containing the fitted data transformer.
* `cleaned_crop_yield.csv`: The finalized dataset after handling missing values and preprocessing.

---

## 📊 Exploratory Data Analysis (EDA)

### 1. Crop Frequency Distribution
An analysis of the top 10 most frequent crops present in our dataset.
![Crop Frequency](crop_frequency_bar_chart.png)

### 2. Rainfall Distribution
Visualizing how rainfall amounts are spread across all recorded regions.
![Rainfall Distribution](rainfall_distribution_histogram.png)

### 3. Rainfall vs. Crop Yield
A scatter plot demonstrating the relationship and trend between precipitation levels and final crop yield.
![Rainfall vs Yield](rainfall_vs_yield_scatter_plot.png)

### 4. Yield Distribution
Analyzing the spread and distribution skewness of the target variable (Crop Yield).
![Yield Distribution](yield_distribution_histogram.png)

---

## 🚀 How to Use the Preprocessor
You can load the saved `.joblib` preprocessor directly into your model training script using:
```python
import joblib

# Load the preprocessing pipeline
preprocessor = joblib.load('preprocessor.joblib')
```
