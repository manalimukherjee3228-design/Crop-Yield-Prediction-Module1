import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def run_module_1(file_path="crop_yield.csv"):
    print("\n--- [Module 1] Starting Single File Preprocessing ---")
    
    # 1. Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found in this folder! Please verify the name.")
        return None
        
    print("1. Loading dataset...")
    df = pd.read_csv(file_path)

    # Display original column names to the terminal for debugging
    print("Columns found in your file:", list(df.columns))

    # Clean column names by removing trailing/leading spaces
    df.columns = df.columns.str.strip()

    # 2. Handle missing data and find the target variable
    print("2. Cleaning missing data...")
    if 'Yield' not in df.columns and 'Production' in df.columns and 'Area' in df.columns:
        df = df.dropna(subset=['Production', 'Area'])
        df = df[(df['Production'] > 0) & (df['Area'] > 0)]
        df['Yield'] = df['Production'] / df['Area']
    elif 'Yield' in df.columns:
        df = df.dropna(subset=['Yield'])
        df = df[df['Yield'] > 0]
    else:
        print("❌ Error: Could not find Yield or Production/Area columns.")
        return None

    # 3. Automatically identifying feature columns matching standard Kaggle datasets
    print("3. Automatically identifying feature columns...")
    possible_categorical = ['Crop', 'Season', 'State', 'Crop_Year']
    possible_numerical = ['Area', 'Annual_Rainfall', 'Rainfall', 'Pesticide', 'Fertilizer']
    
    categorical_cols = [col for col in possible_categorical if col in df.columns]
    numerical_cols = [col for col in possible_numerical if col in df.columns]

    print(f"Using categorical features: {categorical_cols}")
    print(f"Using numerical features: {numerical_cols}")

    # Isolate Features and Target
    X = df[categorical_cols + numerical_cols]
    y = df['Yield']

    # 4. Preprocessing Config (Scaling numbers and Encoding text data)
    print("4. Setting up preprocessing pipeline (Scaling & Encoding)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # 5. Train/Test Split
    print("5. Splitting into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute transformations
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 6. Save configuration for your group
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("✅ Success! 'preprocessor.joblib' saved to folder.")
    print(f"Processed training data shape ready for your team: {X_train_proc.shape}")

    return X_train_proc, X_test_proc, y_train, y_test

if __name__ == "__main__":
    run_module_1()