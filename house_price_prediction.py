
# House Price Prediction using Linear Regression
# SkillCraft Technology - ML Internship Task
# Repository: SCT_ML_1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("🚀 House Price Prediction - Linear Regression")

# Load data (you'll need to upload train.csv)
def load_data():
    """Load the house price dataset"""
    # In your implementation, this loads the CSV
    return pd.read_csv('train.csv')

def preprocess_data(df):
    """Preprocess data for linear regression"""
    # Create total bathrooms
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
    
    # Select features
    X = df[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
    y = df['SalePrice']
    
    return X, y

def train_model(X, y):
    """Train linear regression model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, rmse, r2

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)
    
    # Train model
    model, scaler, rmse, r2 = train_model(X, y)
    
    print(f"✅ Model Training Complete!")
    print(f"📊 R² Score: {r2:.4f}")
    print(f"📊 RMSE: {rmse:.2f}")
    
    # Show coefficients
    print("\n📈 Feature Coefficients:")
    features = ['Square Footage', 'Bedrooms', 'Bathrooms']
    for i, feature in enumerate(features):
        print(f"{feature}: {model.coef_[i]:.2f}")
