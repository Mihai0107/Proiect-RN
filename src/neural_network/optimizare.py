import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# --- 0. PREGATIRE MEDIU ---
# Ne asiguram ca folderele exista, ca sa nu dea eroare la salvare
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('results'): os.makedirs('results')

print("\n🔬 PORNIRE EXPERIMENTE OPTIMIZARE (Grid Search Manual)...")

# --- 1. GENERARE DATE PENTRU OPTIMIZARE ---
# Generam un set nou de date, putin mai mare (6000), special pentru a valida modelele
print("1. Generare date sintetice (6000 mostre)...")
np.random.seed(42) # Seed 42 ca sa avem rezultate reproductibile
n = 6000

data = {
    'Distanța (km)': np.random.uniform(1, 60, n),
    'Nivel trafic': np.random.randint(1, 6, n),
    'Ora livrării': np.random.uniform(8, 22, n),
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n),
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n),
    'Grad de încărcare': np.random.uniform(0.1, 1.0, n)
}
df = pd.DataFrame(data)

# Functia matematica pentru Ground Truth (eticheta corecta)
def calculeaza_timp(row):
    timp = row['Distanța (km)'] * 2.0 + (row['Nivel trafic'] * 5)
    # Penalizari ore varf
    if 8 <= row['Ora livrării'] <= 10 or 16 <= row['Ora livrării'] <= 19: timp += 15
    # Penalizare vineri
    if row['Ziua săptămânii'] == 'Vineri': timp += 10
    # Factor vehicul
    factor = {'Bicicletă': 1.5, 'Scuter': 0.8, 'Camion': 1.3}.get(row['Tip vehicul'], 1.0)
    # Returnam timpul cu putin zgomot (noise)
    return max(5, (timp * factor) + np.random.normal(0, 3))

y = df.apply(calculeaza_timp, axis=1).values

# --- 2. PREPROCESARE ---
print("2. Preprocesare date...")
# Definim transformatorul pentru a converti textul in numere si a scala valorile
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Distanța (km)', 'Nivel trafic', 'Ora livrării', 'Grad de încărcare']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ziua săptămânii', 'Tip vehicul'])
])

X = preprocessor.fit_transform(df)

# Impartim in Train (80%) si Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. DEFINIRE SCENARII EXPERIMENTE ---
# Aici testez diverse configuratii ca sa vad care scoate eroarea cea mai mica
experimente = [
    # Scenariul 1: Configuratia standard (Baseline)
    {"nume": "Exp 1 (Baseline)", "layers": [64, 32], "batch": 32, "lr": 0.001},
    
    # Scenariul 2: Batch size mare (pentru viteza, dar poate pierde precizie)
    {"nume": "Exp 2 (Batch Mare)", "layers": [64, 32], "batch": 128, "lr": 0.001},
    
    # Scenariul 3: Retea mai complexa (Deep Learning mai adanc)
    {"nume": "Exp 3 (Model Complex)", "layers": [128, 64, 32], "batch": 32, "lr": 0.001},
    
    # Scenariul 4: Learning Rate foarte mic (invatare fina)
    {"nume": "Exp 4 (Fine Tuning)", "layers": [64, 32], "batch": 32, "lr": 0.0001}
]

rezultate = []
best_mae = float('inf') # Initializam cea mai buna eroare cu Infinit

print("\n3. Start Rulare Experimente:")
print("="*60)

for exp in experimente:
    print(f"▶️  Rulez: {exp['nume']}...")
    
    # Construire model dinamic in functie de lista 'layers'
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],))) # Input Layer
    
    # Adaugam straturile ascunse din configuratie
    for neurons in exp['layers']:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        
    model.add(tf.keras.layers.Dense(1)) # Output Layer (Regresie)
    
    # Compilare cu Learning Rate variabil
    opt = tf.keras.optimizers.Adam(learning_rate=exp['lr'])
    model.compile(optimizer=opt, loss='mae')
    
    # Antrenare (fara verbose ca sa nu umplem consola)
    history = model.fit(X_train, y_train, epochs=30, batch_size=exp['batch'], validation_split=0.2, verbose=0)
    
    # Evaluare pe setul de test
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"   REZULTAT: MAE (Eroare Medie) = {loss:.4f} minute")
    
    rezultate.append([exp['nume'], loss])
    
    # LOGICA "CHAMPION": Salvam doar daca e cel mai bun de pana acum
    if loss < best_mae:
        best_mae = loss
        print("   🏆 RECORD NOU! Salvam acest model ca fiind cel optimizat.")
        model.save('models/optimized_model.keras')
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print("-" * 60)

# --- 4. SALVARE RAPORT FINAL ---
df_rez = pd.DataFrame(rezultate, columns=['Nume Experiment', 'Eroare Test (MAE)'])
df_rez.to_csv('results/optimization_experiments.csv', index=False)

print("\n✅ Proces Finalizat!")
print(f"Cel mai bun model a avut eroarea {best_mae:.4f} si a fost salvat in 'models/optimized_model.keras'.")
print("\nRezumat Experimente:")
print(df_rez)