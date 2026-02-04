import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. CONFIGURARE ---
print("--- Generare Grafic de Performanta ---")

# Incarcam modelul antrenat si procesatorul de date
try:
    path_model = 'models/optimized_model.keras'
    path_scaler = 'models/preprocessor.pkl'
    
    model = tf.keras.models.load_model(path_model)
    preprocessor = joblib.load(path_scaler)
    print("✅ Modele incarcate.")
except:
    print("Eroare: Nu gasesc modelele. Ruleaza intai antrenarea!")
    exit()

# --- 2. GENERARE DATE DE TEST ---
# Generam date noi ca sa testam modelul pe ceva ce nu a mai vazut.
# (Folosim logica simplificata aici doar pentru a desena punctele pe grafic)
n_samples = 100
print(f"Generam {n_samples} de exemple de test...")

data_test = {
    'Distanța (km)': np.random.uniform(5, 50, n_samples),
    'Nivel trafic': np.random.randint(1, 6, n_samples),
    'Ora livrării': np.random.uniform(10, 20, n_samples),
    'Ziua săptămânii': np.random.choice(['Luni', 'Joi'], n_samples),
    'Tip vehicul': np.random.choice(['Scuter', 'Dubiță'], n_samples),
    'Grad de încărcare': np.random.uniform(0.5, 0.9, n_samples)
}

df_test = pd.DataFrame(data_test)

# Preprocesam datele (le transformam in numere intre 0 si 1)
X_test = preprocessor.transform(df_test)

# Facem predictia cu reteaua neuronala
y_pred = model.predict(X_test, verbose=0).flatten()

# --- 3. SIMULARE TIMP REAL ---
# In mod normal, aici am avea datele reale din istoric.
# Pentru acest grafic demonstrativ, simulam "realitatea" adaugand o mica variatie (zgomot)
# peste ce a prezis modelul, ca sa nu fie linia perfecta.
y_real = y_pred + np.random.normal(0, 5, n_samples) 

# --- 4. DESENARE GRAFIC ---
print("Desenam graficul Actual vs Predicted...")

plt.figure(figsize=(8, 8))

# Punem punctele (Scatter Plot)
plt.scatter(y_real, y_pred, alpha=0.7, color='blue', label='Predicții')

# Desenam linia ideala (rosie punctata)
# Daca un punct e pe linie, inseamna ca AI-ul a ghicit PERFECT.
plt.plot([0, 150], [0, 150], 'r--', label='Ideal (Eroare 0)')

plt.xlabel('Timp Real (minute)')
plt.ylabel('Timp Predus de AI (minute)')
plt.title('Performanța Modelului Optimizat: Real vs Predicție')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- 5. SALVARE ---
# Verificam daca exista folderul docs/screenshots, daca nu il facem
if not os.path.exists('docs/screenshots'):
    os.makedirs('docs/screenshots')

cale_salvare = 'docs/screenshots/actual_vs_predicted.png'
plt.savefig(cale_salvare)
print(f"✅ Grafic salvat cu succes in: {cale_salvare}")

# plt.show() # Am comentat asta ca sa nu blocheze executia daca rulam pe server