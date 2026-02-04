import pandas as pd
import numpy as np
import os

# --- CONFIGURARE INITIALA ---
# Folosim seed 42 pentru a avea mereu aceleasi date cand rulam scriptul (reproductibilitate)
np.random.seed(42)

# Numarul de comenzi simulate. Am ales 5000 pentru a avea suficient material de antrenare
n_samples = 5000

print(f"--- Incepem generarea a {n_samples} de comenzi simulate ---")

# 1. GENERARE CARACTERISTICI (INPUTS)
data = {
    # Distante intre 0.5km si 20km (livrari urbane normale)
    'Distanța (km)': np.round(np.random.uniform(0.5, 20, n_samples), 2),
    
    # Nivel trafic 1-5. Am setat probabilitati inegale pentru ca traficul extrem (1 sau 5) e mai rar
    'Nivel trafic': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
    
    # Orele de program: 07:00 - 22:00
    'Ora livrării': np.round(np.random.uniform(7, 22, n_samples), 2),
    
    # Zilele saptamanii
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n_samples),
    
    # Tipul de vehicul folosit
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n_samples),
    
    # Cat de plin e vehiculul (0.1 = gol, 1.0 = full)
    'Grad de încărcare': np.round(np.random.uniform(0.1, 1.0, n_samples), 2)
}

df = pd.DataFrame(data)

# 2. LOGICA DE CALCUL (SIMULARE FIZICA)
# Viteze medii estimate in oras (in km/h)
speed_map = {
    'Bicicletă': 15, 
    'Scuter': 30, 
    'Dubiță': 45, 
    'Camion': 40  # Camionul e mai lent decat duba
}

# Penalizare de timp in functie de trafic (Multiplicator)
# Trafic 1 (Liber) = 1.0x (timp normal)
# Trafic 5 (Blocat) = 3.0x (de 3 ori mai mult timp)
traffic_penalty = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.0}

def calculeaza_durata(row):
    """
    Simuleaza durata reala bazata pe distanta, viteza si factori externi.
    Formula: (Distanta / Viteza) * Factori + Zgomot Aleator
    """
    viteza = speed_map[row['Tip vehicul']]
    factor_trafic = traffic_penalty[row['Nivel trafic']]
    
    # Calcul fizic de baza (ore -> minute)
    durata_baza = (row['Distanța (km)'] / viteza) * 60
    
    # Ajustam cu traficul si incarcarea (daca e plin, se misca mai greu)
    durata_reala = durata_baza * factor_trafic * (1 + row['Grad de încărcare'] * 0.2)
    
    # Adaugam "Noise" (zgomot) - variatie aleatoare +/- 5 minute
    # pentru ca in realitate apar semafoare, livrari la etaj, etc.
    noise = np.random.normal(0, 5)
    
    # Nu putem avea durate negative, punem minim 5 minute
    return max(5, round(durata_reala + noise))

# Aplicam functia pe fiecare rand
df['Durata estimată (min)'] = df.apply(calculeaza_durata, axis=1)

# 3. INTRODUCERE INTENTIONATA DE "DATĂ MURDARĂ" (NOISE/OUTLIERS)
# Facem asta ca sa avem ce curata in etapa de Preprocessing!

# Stergem random 5% din valorile de trafic (simulam erori de senzor)
df.loc[df.sample(frac=0.05).index, 'Nivel trafic'] = np.nan 

# Introducem 5 valori aberante (150 km distanta in oras) - Outliers
# Acestea ar trebui sa fie detectate si eliminate ulterior
df.loc[df.sample(n=5).index, 'Distanța (km)'] = 150.0 

# 4. SALVARE DATASET
folder_destinatie = 'data/raw'
os.makedirs(folder_destinatie, exist_ok=True)

cale_fisier = os.path.join(folder_destinatie, 'delivery_data.csv')

# Folosim encoding utf-8-sig pentru a se vedea corect diacriticele in Excel
df.to_csv(cale_fisier, index=False, encoding='utf-8-sig')

print(f"✅ Generare completa! Fisier salvat in: {cale_fisier}")
# print(df.head()) # Debug: Verificare primele randuri
# print(df.describe()) # Debug: Verificare statistici