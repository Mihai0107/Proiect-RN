import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- CONFIGURARE ---
print("--- Incepem Analiza Erorilor (Stress Test) ---")

# 1. INCARCARE MODEL ANTRENAT
# Avem nevoie de modelul optimizat si de scaler-ul folosit la antrenare
try:
    print("⏳ Incarcare model si procesator...")
    model = tf.keras.models.load_model('models/optimized_model.keras')
    preprocessor = joblib.load('models/preprocessor.pkl')
except Exception as e:
    print(f"Eroare: Nu gasesc fisierele. Ai rulat antrenarea? {e}")
    exit()

# 2. GENERARE DATE DE TEST (SIMULARE NOUA)
# Generam 500 de comenzi noi, pe care modelul NU le-a vazut niciodata
# Folosim un seed fix (123) ca sa avem mereu aceleasi "erori" de prezentat profesorului
np.random.seed(123) 
n = 500

print(f"Generam {n} de situatii noi de livrare...")

data = {
    'Distanța (km)': np.random.uniform(1, 60, n), # Testam si distante mai mari sa vedem daca se sperie
    'Nivel trafic': np.random.randint(1, 6, n),
    'Ora livrării': np.random.uniform(8, 22, n),
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n),
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n),
    'Grad de încărcare': np.random.uniform(0.1, 1.0, n)
}
df = pd.DataFrame(data)

# 3. CALCUL TIMP "REAL" (Ground Truth)
# Aici folosim aceeasi logica matematica din scriptul de generare_date.py
# Vrem sa stim cat ar trebui sa dureze CORECT, ca sa comparam cu ce zice AI-ul.
def calculeaza_timp_real_simulat(row):
    # Formula aproximativa folosita la generare:
    # Timp baza = distanta * factor + trafic
    timp = row['Distanța (km)'] * 2.0 + (row['Nivel trafic'] * 5)
    
    # Penalizari ore de varf
    if 8 <= row['Ora livrării'] <= 10 or 16 <= row['Ora livrării'] <= 19: 
        timp += 15
        
    # Vinerea e mai aglomerat
    if row['Ziua săptămânii'] == 'Vineri': 
        timp += 10
        
    # Factor vehicul
    factor_vehicul = {'Bicicletă': 1.5, 'Scuter': 0.8, 'Camion': 1.3}.get(row['Tip vehicul'], 1.0)
    
    # Adaugam zgomotul aleatoriu (neprevazutul din trafic)
    noise = np.random.normal(0, 5) 
    
    return max(5, (timp * factor_vehicul) + noise)

# Calculam coloana de referinta
y_real = df.apply(calculeaza_timp_real_simulat, axis=1).values

# 4. PREDICTIA AI
# Transformam datele exact cum am facut la antrenare
X_input = preprocessor.transform(df)

# Punem modelul sa ghiceasca
# verbose=0 ca sa nu umplem consola
y_pred = model.predict(X_input, verbose=0).flatten()

# 5. ANALIZA COMPARATIVA
df['Timp Real'] = y_real
df['Timp Predus'] = y_pred

# Calculam cat de mult a gresit (Eroarea Absoluta)
df['Eroare (Absolută)'] = abs(df['Timp Real'] - df['Timp Predus'])
df['Diferenta'] = df['Timp Real'] - df['Timp Predus'] # Pozitiv = A subestimat, Negativ = A supraestimat

# 6. RAPORTARE FINALĂ
# Sortam descrescator dupa eroare ca sa vedem cele mai mari "gafe"
top_erori = df.nlargest(5, 'Eroare (Absolută)')

print("\n" + "="*80)
print("TOP 5 CELE MAI MARI ERORI ALE MODELULUI")
print("(Aceste date le voi folosi in README pentru a explica limitele sistemului)")
print("="*80)

for index, row in top_erori.iterrows():
    print(f"INDEX TEST: #{index}")
    print(f"SCENARIU: {row['Tip vehicul']} pe {row['Distanța (km)']:.1f}km | Trafic {row['Nivel trafic']} | Ora {row['Ora livrării']:.1f}")
    print(f"REZULTAT:  Realitate = {row['Timp Real']:.1f} min  VS  AI = {row['Timp Predus']:.1f} min")
    
    # Mesaj personalizat in functie de eroare
    if row['Diferenta'] > 0:
        print(f"⚠️  CONCLUZIE: AI-ul a fost prea optimist (a zis cu {abs(row['Diferenta']):.1f} min mai putin)")
    else:
        print(f"⚠️  CONCLUZIE: AI-ul a fost prea pesimist (a zis cu {abs(row['Diferenta']):.1f} min mai mult)")
        
    print("-" * 40)