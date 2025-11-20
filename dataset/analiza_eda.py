import pandas as pd

df = pd.read_csv('delivery_data.csv')

print("--- 1. DIMENSIUNI SET DATE ---")
print(f"Număr observații (rânduri): {df.shape[0]}")
print(f"Număr caracteristici (coloane): {df.shape[1]}")

print("\n--- 2. STATISTICI DESCRIPTIVE (Media, Min, Max) ---")
print(df.describe().round(2))

print("\n--- 3. VALORI LIPSĂ (Ce trebuie să curățăm) ---")
print(df.isnull().sum())

print("\n--- 4. CORELAȚIA CU DURATA (Ce influențează livrarea?) ---")
numeric_df = df.select_dtypes(include=['number'])
corelatie = numeric_df.corr()['Durata estimată (min)'].sort_values(ascending=False)
print(corelatie)