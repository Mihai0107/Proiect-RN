import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Configurare
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('results'): os.makedirs('results')

print("🔬 PORNIRE EXPERIMENTE OPTIMIZARE...\n")

# 1. GENERARE DATE (Aceleași ca înainte)
n = 6000 # Mărim puțin setul de date pentru optimizare
np.random.seed(42)
data = {
    'Distanța (km)': np.random.uniform(1, 60, n),
    'Nivel trafic': np.random.randint(1, 6, n),
    'Ora livrării': np.random.uniform(8, 22, n),
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n),
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n),
    'Grad de încărcare': np.random.uniform(0.1, 1.0, n)
}
df = pd.DataFrame(data)

def calculeaza_timp(row):
    timp = row['Distanța (km)'] * 2.0 + (row['Nivel trafic'] * 5)
    if 8 <= row['Ora livrării'] <= 10 or 16 <= row['Ora livrării'] <= 19: timp += 15
    if row['Ziua săptămânii'] == 'Vineri': timp += 10
    factor = {'Bicicletă': 1.5, 'Scuter': 0.8, 'Camion': 1.3}.get(row['Tip vehicul'], 1.0)
    return max(5, (timp * factor) + np.random.normal(0, 3))

y = df.apply(calculeaza_timp, axis=1).values

# 2. PREPROCESARE
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Distanța (km)', 'Nivel trafic', 'Ora livrării', 'Grad de încărcare']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ziua săptămânii', 'Tip vehicul'])
])
X = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINIRE EXPERIMENTE
experimente = [
    {"nume": "Exp 1 (Baseline)", "layers": [64, 32], "batch": 32, "lr": 0.001},
    {"nume": "Exp 2 (Batch Mare)", "layers": [64, 32], "batch": 128, "lr": 0.001},
    {"nume": "Exp 3 (Model Complex)", "layers": [128, 64, 32], "batch": 32, "lr": 0.001},
    {"nume": "Exp 4 (Learning Rate Mic)", "layers": [64, 32], "batch": 32, "lr": 0.0001}
]

rezultate = []
best_mae = float('inf')

for exp in experimente:
    print(f"--- Rulare {exp['nume']} ---")
    
    # Construire model dinamic
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for neurons in exp['layers']:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1)) # Output layer
    
    opt = tf.keras.optimizers.Adam(learning_rate=exp['lr'])
    model.compile(optimizer=opt, loss='mae')
    
    # Antrenare
    history = model.fit(X_train, y_train, epochs=30, batch_size=exp['batch'], validation_split=0.2, verbose=0)
    
    # Evaluare
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"   MAE (Eroare): {loss:.4f}")
    
    rezultate.append([exp['nume'], loss])
    
    # Salvare cel mai bun model
    if loss < best_mae:
        best_mae = loss
        model.save('models/optimized_model.keras')
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        print("   🏆 New Best Model Saved!")

# Salvare tabel rezultate
df_rez = pd.DataFrame(rezultate, columns=['Experiment', 'MAE Test'])
df_rez.to_csv('results/optimization_experiments.csv', index=False)
print("\n✅ Optimizare Finalizată! Cel mai bun model este în 'models/optimized_model.keras'")
print(df_rez)