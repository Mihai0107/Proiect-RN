import pandas as pd

# Incarcam setul de date brut pe care l-am generat in pasul anterior
# Este important sa verificam daca s-a incarcat corect
df = pd.read_csv('data/raw/delivery_data.csv')

print("--- PASUL 1: Verificare Dimensiuni Dataset ---")
# Vreau sa vad cate exemple (linii) si cate caracteristici (coloane) avem in total
print(f"Număr observații (linii): {df.shape[0]}")
print(f"Număr caracteristici (coloane): {df.shape[1]}")

print("\n--- PASUL 2: Statistici Descriptive (Min, Max, Medii) ---")
# Folosesc describe() pentru a vedea distributia datelor.
# Rotunjesc la 2 zecimale ca sa fie tabelul citibil in consola.
print(df.describe().round(2))

print("\n--- PASUL 3: Verificare Valori Lipsă (Null/NaN) ---")
# Verificam daca avem gauri in date.
# Nota: Ar trebui sa vedem lipsuri la 'Nivel trafic' (le-am introdus intentionat ca zgomot)
print(df.isnull().sum())

print("\n--- PASUL 4: Analiza Corelației (Ce influentează durata?) ---")
# Selectam doar coloanele numerice pentru ca functia corr() nu merge pe text
numeric_df = df.select_dtypes(include=['number'])

# Calculam corelatia tuturor coloanelor cu 'Durata estimată'
# Sortam descrescator: vrem sa vedem sus factorii cu cel mai mare impact (pozitiv sau negativ)
corelatie = numeric_df.corr()['Durata estimată (min)'].sort_values(ascending=False)
print(corelatie)