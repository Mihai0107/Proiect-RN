import pandas as pd


df = pd.read_csv('data/raw/delivery_data.csv')

print("--- 1. DIMENSIUNI SET DATE ---")
print(f"Număr observații: {df.shape[0]}")
print(f"Număr caracteristici: {df.shape[1]}")

print("\n--- 2. STATISTICI DESCRIPTIVE ---")
print(df.describe().round(2))

print("\n--- 3. VALORI LIPSĂ ---")
print(df.isnull().sum())

print("\n--- 4. CORELAȚIA CU DURATA ---")
numeric_df = df.select_dtypes(include=['number'])
corelatie = numeric_df.corr()['Durata estimată (min)'].sort_values(ascending=False)
print(corelatie)