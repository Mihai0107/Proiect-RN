import pandas as pd
import numpy as np
import os


np.random.seed(42)
n_samples = 5000


data = {
    'Distanța (km)': np.round(np.random.uniform(0.5, 50, n_samples), 2),
    'Nivel trafic': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
    'Ora livrării': np.round(np.random.uniform(7, 22, n_samples), 2),
    'Ziua săptămânii': np.random.choice(['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'], n_samples),
    'Tip vehicul': np.random.choice(['Bicicletă', 'Scuter', 'Dubiță', 'Camion'], n_samples),
    'Grad de încărcare': np.round(np.random.uniform(0.1, 1.0, n_samples), 2)
}
df = pd.DataFrame(data)


speed_map = {'Bicicletă': 15, 'Scuter': 30, 'Dubiță': 45, 'Camion': 40}
traffic_penalty = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.0}

def calculeaza_durata(row):
    viteza = speed_map[row['Tip vehicul']]
    factor_trafic = traffic_penalty[row['Nivel trafic']]
    durata_baza = (row['Distanța (km)'] / viteza) * 60
    durata_reala = durata_baza * factor_trafic * (1 + row['Grad de încărcare'] * 0.2)
    noise = np.random.normal(0, 5)
    return max(5, round(durata_reala + noise))

df['Durata estimată (min)'] = df.apply(calculeaza_durata, axis=1)


df.loc[df.sample(frac=0.05).index, 'Nivel trafic'] = np.nan 
df.loc[df.sample(n=5).index, 'Distanța (km)'] = 150.0 


os.makedirs('data/raw', exist_ok=True)


df.to_csv('data/raw/delivery_data.csv', index=False, encoding='utf-8-sig')

print("✅ Fișierul a fost salvat în 'data/raw/delivery_data.csv' cu diacritice corecte!")
print(df.head())