import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

# Configurare directoare conform Etapei 5
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# 1. Încărcare date
print("1. Încărcare date...")
try:
    X_train = np.load('data/train/X_train.npy')
    y_train = np.load('data/train/y_train.npy')
    X_val = np.load('data/validation/X_val.npy')
    y_val = np.load('data/validation/y_val.npy')
    X_test = np.load('data/test/X_test.npy')
    y_test = np.load('data/test/y_test.npy')
except FileNotFoundError:
    print("EROARE: Nu găsesc fișierele .npy! Rulează preprocesare_finala.py.")
    exit()

# 2. Definire Model (Arhitectura din Etapa 4)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(32, activation='relu'),
    Dense(1) # Activare lineara (implicit) pentru regresie
])

# Compilare - Folosim MAE pentru interpretare umana
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Antrenare cu Callbacks (Cerința Nivel 1 & 2)
print("\n2. Începe antrenarea...")

# Early Stopping (Nivel 2)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# CSV Logger (OBLIGATORIU Etapa 5)
csv_logger = CSVLogger('results/training_history.csv')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, csv_logger],
    verbose=1
)

# 4. Salvare Model Antrenat (OBLIGATORIU)
model.save('models/trained_model.keras') # Folosim extensia moderna .keras
print("\n💾 Model salvat în 'models/trained_model.keras'")

# 5. Evaluare și Salvare Metrici (OBLIGATORIU)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

# Calculăm și R2 Score (Metrică bună pentru regresie)
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

metrics = {
    "test_mae": round(test_mae, 4),
    "test_mse": round(test_loss, 4),
    "test_r2_score": round(r2, 4)
}

with open('results/test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("📝 Metrici salvate în 'results/test_metrics.json'")

# 6. Generare Grafic Loss (OBLIGATORIU Nivel 2)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Curba de Învățare (Loss)')
plt.xlabel('Epoci')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('docs/loss_curve.png')
print("nt📈 Grafic salvat în 'docs/loss_curve.png'")

print(f"\n✅ REZULTAT FINAL PE TEST: MAE = {test_mae:.2f} minute")