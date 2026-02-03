import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. Încărcăm modelul și procesatorul
print("⏳ Încărcare model...")
model = tf.keras.models.load_model('models/optimized_model.keras')
preprocessor = joblib.load('models/preprocessor.pkl')

# 2. Generăm date de test (500 de exemple noi)
np.random.seed(123) # Folosim un seed fix ca să avem aceleași rezultate
n = 500
data = {
    'Distanța (km)': np.random.uniform(1, 60, n),
    'Nivel trafic': np.random.randint(1, 6, n),
    'Ora livrării': np.random.uniform(8, 22, n),
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n),
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n),
    'Grad de încărcare': np.random.uniform(0.1, 1.0, n)
}
df = pd.DataFrame(data)

# 3. Calculăm Timpul REAL (Formula matematică originală)
def calculeaza_timp_real(row):
    timp = row['Distanța (km)'] * 2.0 + (row['Nivel trafic'] * 5)
    if 8 <= row['Ora livrării'] <= 10 or 16 <= row['Ora livrării'] <= 19: timp += 15
    if row['Ziua săptămânii'] == 'Vineri': timp += 10
    factor = {'Bicicletă': 1.5, 'Scuter': 0.8, 'Camion': 1.3}.get(row['Tip vehicul'], 1.0)
    # Adăugăm zgomot aleatoriu (neprevăzutul)
    noise = np.random.normal(0, 5) 
    return max(5, (timp * factor) + noise)

y_real = df.apply(calculeaza_timp_real, axis=1).values

# 4. Facem Predicția cu Modelul AI
X_input = preprocessor.transform(df)
y_pred = model.predict(X_input, verbose=0).flatten()

# 5. Calculăm diferența (Eroarea)
df['Timp Real'] = y_real
df['Timp Predus'] = y_pred
df['Eroare (Absolută)'] = abs(df['Timp Real'] - df['Timp Predus'])
df['Diferenta'] = df['Timp Real'] - df['Timp Predus'] # Cu minus sau plus

# 6. Sortăm și afișăm cele mai mari 5 erori
top_erori = df.nlargest(5, 'Eroare (Absolută)')

print("\n" + "="*80)
print("TOP 5 ERORI (Copiaza datele astea in README)")
print("="*80)

for index, row in top_erori.iterrows():
    print(f"INDEX: #{index}")
    print(f"Scenariu: Dist: {row['Distanța (km)']:.1f}km | Trafic: {row['Nivel trafic']} | Vehicul: {row['Tip vehicul']} | Ora: {row['Ora livrării']:.1f}")
    print(f"REZULTAT: Real = {row['Timp Real']:.1f} min  |  Predus AI = {row['Timp Predus']:.1f} min")
    print(f"EROARE: {row['Diferenta']:.1f} minute (AI a greșit cu atât)")
    print("-" * 40)