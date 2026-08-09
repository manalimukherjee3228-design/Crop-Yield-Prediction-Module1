import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def run_module_1(file_path="crop_yield.csv"):
    print("\n--- STEP 1: DATASET COLLECTION ---")
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print("\n--- STEP 2 & 3: EXPLORATION & CLEANING ---")
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Step 4 & 5 Rule Validation
    if 'Yield' not in df.columns and 'Production' in df.columns and 'Area' in df.columns:
        df = df.dropna(subset=['Production', 'Area'])
        df = df[(df['Production'] >= 0) & (df['Area'] > 0)]
        df['Yield'] = df['Production'] / df['Area']
    elif 'Yield' in df.columns:
        df = df.dropna(subset=['Yield'])
        df = df[df['Yield'] >= 0]

    rainfall_col = next((c for c in df.columns if 'rain' in c.lower()), None)
    if rainfall_col:
        df = df.dropna(subset=[rainfall_col])
        df = df[df[rainfall_col] >= 0]

    # Save Cleaned CSV
    df.to_csv("cleaned_crop_yield.csv", index=False)
    print("💾 Saved 'cleaned_crop_yield.csv'")

    # ==========================================
    # NEW STEP 6: AUTOMATIC EDA CHARTS GENERATION
    # ==========================================
    print("\n--- STEP 6: GENERATING EDA CHARTS ---")
    os.makedirs("eda_charts", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: Crop Frequency
    if 'Crop' in df.columns:
        plt.figure(figsize=(10, 5))
        df['Crop'].value_counts().head(10).plot(kind='bar', color='skyblue')
        plt.title("Top 10 Most Frequent Crops")
        plt.ylabel("Record Count")
        plt.xticks(rotation=45)
        plt.savefig("eda_charts/crop_frequency.png", bbox_inches='tight')
        plt.close()

    # Chart 2: Rainfall Distribution
    if rainfall_col:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[rainfall_col], kde=True, color='green')
        plt.title("Rainfall Distribution Map")
        plt.savefig("eda_charts/rainfall_distribution.png", bbox_inches='tight')
        plt.close()

    # Chart 3: Yield Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Yield'], kde=True, color='gold')
    plt.title("Crop Yield Variance Map")
    plt.savefig("eda_charts/yield_distribution.png", bbox_inches='tight')
    plt.close()

    # Chart 4: Rainfall vs Yield Relationship
    if rainfall_col:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df.sample(min(1000, len(df))), x=rainfall_col, y='Yield', alpha=0.5, color='teal')
        plt.title("Rainfall vs Crop Yield Relationship")
        plt.savefig("eda_charts/rainfall_vs_yield.png", bbox_inches='tight')
        plt.close()

    print("📊 SUCCESS: All 4 analysis charts saved in the 'eda_charts/' folder!")

    # Machine Learning Processing Split
    possible_categorical = ['Crop', 'Season', 'State', 'Crop_Year']
    possible_numerical = ['Area', 'Annual_Rainfall', 'Rainfall', 'Pesticide', 'Fertilizer']
    categorical_cols = [col for col in possible_categorical if col in df.columns]
    numerical_cols = [col for col in possible_numerical if col in df.columns]

    X = df[categorical_cols + numerical_cols]
    y = df['Yield']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "preprocessor.joblib")
    print("✅ Pipeline preprocessor setup saved successfully!")
    return X_train_proc, X_test_proc, y_train, y_test

if __name__ == "__main__":
    run_module_1()
