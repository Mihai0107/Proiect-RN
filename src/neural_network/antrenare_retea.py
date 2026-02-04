import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.metrics import r2_score

# --- 0. CONFIGURARE ---
# Ne asiguram ca avem unde sa salvam rezultatele
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('docs', exist_ok=True) # Aici vom pune graficele

print("\n🚀 START PROCES DE ANTRENARE...")

# --- 1. INCARCARE DATE PROCESATE ---
print("1. Incarcam datele transformate in numpy arrays...")
try:
    # Incarcam datele salvate la pasul anterior (preprocesare)
    X_train = np.load('data/train/X_train.npy')
    y_train = np.load('data/train/y_train.npy')
    X_val = np.load('data/validation/X_val.npy')
    y_val = np.load('data/validation/y_val.npy')
    X_test = np.load('data/test/X_test.npy')
    y_test = np.load('data/test/y_test.npy')
    
    # print(f"Shape X_train: {X_train.shape}") # Debug: Verificare dimensiuni
except FileNotFoundError:
    print("❌ EROARE CRITICA: Nu gasesc fisierele .npy!")
    print("Sfat: Ruleaza mai intai scriptul 'preprocesare_finala.py'!")
    exit()

# --- 2. DEFINIRE ARHITECTURA MODEL ---
# Folosim o retea Feed-Forward simpla (MLP)
model = Sequential([
    # Stratul de intrare + Primul strat ascuns (64 neuroni)
    # input_shape este necesar doar la primul strat
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    
    # Al doilea strat ascuns (32 neuroni) - reducem complexitatea treptat
    Dense(32, activation='relu'),
    
    # Stratul de iesire (1 singur neuron)
    # Nu avem functie de activare (sau linear) pentru ca facem REGRESIE (prezicem un numar continuu)
    Dense(1) 
])

# --- 3. COMPILARE ---
# Optimizer: Adam (cel mai bun generalist)
# Loss: MSE (Mean Squared Error) - penalizeaza erorile mari
# Metrics: MAE (Mean Absolute Error) - e mai usor de citit de oameni (eroarea in minute)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 4. PREGATIRE CALLBACKS ---
print("\n2. Configuram monitorizarea antrenarii...")

# EarlyStopping: Oprim antrenarea daca modelul nu mai invata timp de 10 epoci
# Asta previne Overfitting-ul si economiseste timp
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# CSVLogger: Salvam istoricul antrenarii intr-un fisier excel/csv pentru grafice
csv_logger = CSVLogger('results/training_history.csv')

# --- 5. ANTRENAREA PROPRIU-ZISA ---
print("-> Incepem antrenarea (maxim 100 epoci)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32, # Actualizam greutatile la fiecare 32 de exemple
    callbacks=[early_stop, csv_logger],
    verbose=1 # Afisam bara de progres
)

# --- 6. SALVARE MODEL ---
# Salvam modelul final in formatul nou Keras
model.save('models/trained_model.keras') 
print("\n💾 Modelul antrenat a fost salvat in 'models/trained_model.keras'")

# --- 7. EVALUARE FINALA PE TEST ---
print("3. Evaluam performanta pe datele de TEST (necunoscute)...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

# Calculam si R2 Score (cat de bine "se potriveste" modelul pe date)
y_pred = model.predict(X_test, verbose=0)
r2 = r2_score(y_test, y_pred)

# Salvam metricile intr-un JSON frumos formatat
metrics = {
    "test_mae": round(test_mae, 4),      # Eroarea medie in minute
    "test_mse": round(test_loss, 4),     # Eroarea patratica
    "test_r2_score": round(r2, 4)        # Scorul R2 (aproape de 1 e perfect)
}

with open('results/test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("📝 Metricile finale au fost salvate in 'results/test_metrics.json'")

# --- 8. GENERARE GRAFIC (Loss Curve) ---
# Acest grafic va ajunge in documentatie
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eroare Antrenare (Train Loss)')
plt.plot(history.history['val_loss'], label='Eroare Validare (Val Loss)')
plt.title('Curba de Învățare (Evoluția Erorii)')
plt.xlabel('Epoci')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

path_grafic = 'docs/loss_curve.png'
plt.savefig(path_grafic)
print(f"📈 Graficul a fost salvat in '{path_grafic}'")

print(f"\n✅ GATA! Eroare medie finală (MAE): {test_mae:.2f} minute")