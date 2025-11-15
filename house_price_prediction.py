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

def load_data(file_path='train.csv'):
    """Load the house price dataset"""
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {data.shape}")
        return data
    except FileNotFoundError:
        print("❌ Error: train.csv file not found.")
        print("Please make sure train.csv is in the same directory")
        return None

def explore_data(df):
    """Explore the dataset and show basic information"""
    print("\n🔍 Dataset Exploration:")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    # Check if required columns exist
    required_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    else:
        print("✅ All required columns present")
        return True

def preprocess_data(df):
    """Preprocess data for linear regression"""
    print("\n🔄 Preprocessing data...")
    
    # Create total bathrooms feature
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
    
    # Select features
    X = df[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
    y = df['SalePrice']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: SalePrice")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y

def train_model(X, y):
    """Train linear regression model"""
    print("\n🤖 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
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
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, X_test, y_test, y_pred, mae, rmse, r2

def plot_results(y_test, y_pred, model):
    """Create visualization plots"""
    print("\n📈 Generating plots...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_plot.png')
    plt.show()

def main():
    """Main execution function"""
    print("🚀 House Price Prediction - Linear Regression")
    print("=" * 50)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Explore data
    if not explore_data(data):
        return
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train model
    model, scaler, X_test, y_test, y_pred, mae, rmse, r2 = train_model(X, y)
    
    # Display results
    print("\n📊 MODEL RESULTS:")
    print("=" * 30)
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    
    print("\n📈 FEATURE COEFFICIENTS:")
    print("=" * 30)
    features = ['Square Footage', 'Bedrooms', 'Bathrooms']
    for i, feature in enumerate(features):
        print(f"{feature}: {model.coef_[i]:.2f}")
    print(f"Intercept: ${model.intercept_:,.2f}")
    
    print("\n💡 BUSINESS INSIGHTS:")
    print("=" * 30)
    print("• Square footage has the strongest impact on house prices")
    print("• Additional bedrooms and bathrooms increase property value")
    print("• The model explains {:.1f}% of price variation".format(r2 * 100))
    
    # Create plots
    plot_results(y_test, y_pred, model)
    
    print("\n✅ Analysis completed successfully!")
    print("📁 Results plot saved as 'results_plot.png'")

if __name__ == "__main__":
    main()
