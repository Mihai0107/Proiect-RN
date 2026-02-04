import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib # Folosit pentru a salva "reteta" de preprocesare (scaler-ul)
import os

# --- 1. INCARCARE DATE BRUTE ---
# Citim CSV-ul generat anterior
df = pd.read_csv('data/raw/delivery_data.csv')
print("1. Date incarcate cu succes.")

# --- 2. CURATARE DATE (DATA CLEANING) ---

# A. Tratarea valorilor lipsa (Missing Values)
# La generare am sters intentionat unele valori de trafic.
# Le inlocuim cu Mediana (e mai robusta decat Media la valori extreme).
mediana_trafic = df['Nivel trafic'].median()
df['Nivel trafic'] = df['Nivel trafic'].fillna(mediana_trafic)

# B. Eliminarea Outlierilor (Valori extreme)
# Scoatem acele livrari de 150km facute cu bicicleta (erori de generare)
# Pastram doar 99% din date (eliminam top 1% cele mai mari distante)
limita_superioara = df['Distanța (km)'].quantile(0.99)
df = df[df['Distanța (km)'] <= limita_superioara]

# --- 3. IMPARTIRE SETURI DE DATE (SPLIT) ---
X = df.drop('Durata estimată (min)', axis=1) # Inputuri
y = df['Durata estimată (min)']              # Ce vrem sa prezicem (Target)

# Impartim in 3 seturi: Train (Antrenare), Validation (Validare), Test (Testare Finala)
# Pas 1: Separam 15% pentru Test Final
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Pas 2: Din ce a ramas (X_temp), mai rupem o bucata pentru Validare
# (aproximativ 15% din totalul initial)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

# --- 4. PREPROCESARE (SCALARE SI CODARE) ---
# Definim coloanele numerice si cele text
numeric_features = ['Distanța (km)', 'Nivel trafic', 'Ora livrării', 'Grad de încărcare']
categorical_features = ['Ziua săptămânii', 'Tip vehicul']

# Construim "Pipeline"-ul de transformare
preprocessor = ColumnTransformer(
    transformers=[
        # Numerele le scalam (StandardScaler) ca sa aiba media 0 si deviatia 1
        ('num', StandardScaler(), numeric_features),
        # Textul il transformam in vectori de 0 si 1 (OneHot)
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. APLICARE TRANSFORMARI ---
# ATENTIE: Folosim fit_transform DOAR pe datele de antrenare!
# Reteaua trebuie sa invete media si deviatia doar din Train, nu din Test (ar fi trisare/Data Leakage)
X_train_processed = preprocessor.fit_transform(X_train)

# Pe Validation si Test doar aplicam regulile invatate (transform)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Convertim si target-ul (y) in numpy arrays simple pentru Keras
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

# --- 6. SALVARE DATE PROCESATE ---
# Facem folderele daca nu exista
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Salvam matricele procesate (.npy e format binar rapid)
np.save('data/train/X_train.npy', X_train_processed)
np.save('data/train/y_train.npy', y_train)
np.save('data/validation/X_val.npy', X_val_processed)
np.save('data/validation/y_val.npy', y_val)
np.save('data/test/X_test.npy', X_test_processed)
np.save('data/test/y_test.npy', y_test)

# FOARTE IMPORTANT: Salvam obiectul 'preprocessor' ca sa il folosim in aplicatia finala (.exe)
# Altfel aplicatia nu ar sti cum sa scaleze datele introduse de utilizator
joblib.dump(preprocessor, 'models/preprocessor.pkl') 

print("✅ Succes! Datele sunt gata de antrenare si scaler-ul a fost salvat.")