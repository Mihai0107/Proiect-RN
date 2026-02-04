import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import json

# --- 1. CONFIGURARE ---
# Folosesc calea relativa pentru a nu avea probleme daca mut proiectul pe alt PC
PROJECT_ROOT = os.getcwd() 

print(f"\n🚀 PORNIRE EXPERIMENT BONUS: Arhitectură Complexă vs Standard...")

# --- 2. INCARCARE DATE ---
# Incarcam datele procesate (numpy arrays) pe care le-am salvat la pasul de preprocesare
try:
    X_train = np.load(os.path.join(PROJECT_ROOT, 'data', 'train', 'X_train.npy'))
    y_train = np.load(os.path.join(PROJECT_ROOT, 'data', 'train', 'y_train.npy'))
    X_val = np.load(os.path.join(PROJECT_ROOT, 'data', 'validation', 'X_val.npy'))
    y_val = np.load(os.path.join(PROJECT_ROOT, 'data', 'validation', 'y_val.npy'))
    X_test = np.load(os.path.join(PROJECT_ROOT, 'data', 'test', 'X_test.npy'))
    y_test = np.load(os.path.join(PROJECT_ROOT, 'data', 'test', 'y_test.npy'))
    print("✅ Date incarcate corect.")
except FileNotFoundError:
    print("❌ EROARE: Nu găsesc datele .npy! Ruleaza intai scriptul de preprocesare.")
    exit()

# --- 3. DEFINIRE MODEL BONUS (Deep Neural Network) ---
# Am vrut sa testez daca o retea mai adanca (mai multe straturi) si mai lata (mai multi neuroni)
# poate sa prinda pattern-uri mai complexe decat modelul simplu.

# Modelul Standard avea: 64 -> 32 -> 1
# Modelul Acesta are:   128 -> 64 -> 32 -> 1
model_bonus = Sequential([
    # Strat intrare mai mare (128 neuroni)
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    
    # Am adaugat Dropout(0.2) pentru a preveni Overfitting-ul.
    # Asta inseamna ca dezactivam random 20% din neuroni la fiecare pas de antrenare
    # ca reteaua sa nu memoreze datele pe de rost.
    Dropout(0.2), 
    
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    
    # Strat de iesire (1 neuron = timpul estimat)
    Dense(1)
])

model_bonus.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 4. ANTRENARE ---
# Folosesc EarlyStopping ca sa nu pierd timp daca modelul nu mai invata nimic nou
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(" -> Incepem antrenarea modelului complex (poate dura putin mai mult)...")
history = model_bonus.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0 # Nu afisam toate liniile ca sa tinem consola curata
)

# --- 5. EVALUARE SI COMPARATIE ---
# Calculam eroarea pe setul de test (date noi)
mae_bonus = model_bonus.evaluate(X_test, y_test, verbose=0)[1]

# Citesc rezultatele modelului standard (cel simplu) din fisierul salvat anterior
# ca sa pot face o comparatie directa
mae_standard = 6.45 # Valoare default in caz ca nu gasim fisierul
try:
    path_metrics = os.path.join(PROJECT_ROOT, 'results', 'test_metrics.json')
    if os.path.exists(path_metrics):
        with open(path_metrics, 'r') as f:
            metrics_standard = json.load(f)
            mae_standard = metrics_standard['test_mae']
except Exception as e:
    print(f"Nota: Nu am putut incarca istoricul vechi ({e}), folosim valoarea de referinta.")

print("\n" + "="*50)
print("REZULTATE COMPARATIVE: SIMPLU vs COMPLEX")
print("="*50)
print(f"1. Model Standard (64->32):      MAE = {mae_standard:.4f} min")
print(f"2. Model Bonus (128->64->32+Drop): MAE = {mae_bonus:.4f} min")
print("-" * 50)

# Decizie finala
if mae_bonus < mae_standard:
    diff = mae_standard - mae_bonus
    print(f"✅ CONCLUZIE: Modelul complex a castigat! A redus eroarea cu {diff:.4f} minute.")
    
    # Salvam acest model ca "bonus_model.keras"
    save_path = os.path.join(PROJECT_ROOT, 'models', 'bonus_model.keras')
    model_bonus.save(save_path)
    print(f"💾 Am salvat modelul imbunatatit in: models/bonus_model.keras")
else:
    print("⚠️ CONCLUZIE: Modelul complex NU a adus imbunatatiri semnificative.")
    print("   Cauza posibila: Datele sunt prea simple pentru o retea asa mare (Overkill).")
    print("   Ramanem la Modelul Standard ca fiind cel optim.")
    
print("="*50)