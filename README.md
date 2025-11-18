# House Price Prediction using Linear Regression

## 📋 Project Overview
This project implements a linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms.

## 🎯 Task Description
**Organization**: SkillCraft Technology  
**Track**: Machine Learning  
**Task**: Linear Regression Implementation  
**Repository**: SCT_ML_1

## 📊 Dataset
- **Source**: House Prices - Advanced Regression Techniques (Kaggle)
- **Features Used**:
  - GrLivArea: Above grade living area square feet
  - BedroomAbvGr: Number of bedrooms above grade
  - FullBath: Number of full bathrooms
  - HalfBath: Number of half bathrooms

## 🛠️ Implementation
### Model: Linear Regression
### Features:
1. Square footage (GrLivArea)
2. Number of bedrooms (BedroomAbvGr) 
3. Total bathrooms (FullBath + 0.5 * HalfBath)

### Steps:
1. Data loading and exploration
2. Feature engineering
3. Data preprocessing
4. Model training
5. Model evaluation
6. Results visualization

## 📈 Results
- **R² Score**: 0.5476
- **RMSE**: $57,243.20
- **Key Insights**: Square footage showed the strongest correlation with house prices

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python house_price_prediction.py
SCT_ML_1/
├── README.md
├── house_price_prediction.py
├── house_price_linear_regression.ipynb
├── requirements.txt
└── results_summary.md
Ramitha
Machine Learning Intern - SkillCraft Technology
