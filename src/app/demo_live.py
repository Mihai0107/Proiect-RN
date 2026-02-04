import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# --- IMPORTURI PENTRU COMPATIBILITATE ---
# Chiar daca nu le folosim direct aici, joblib are nevoie de ele ca sa stie
# cum sa reconstruiasca obiectul 'preprocessor.pkl' incarcat.
import sklearn.compose 
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.tree
import sklearn.utils

# Ascundem mesajele de debug de la TensorFlow (ca sa nu sperie utilizatorul in consola)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n🚀 PORNIRE SISTEM DE PREDICȚIE LIVE (MOD CONSOLĂ)...")

# --- INCARCARE RESURSE ---
try:
    # Aici incarcam modelul antrenat (.keras) si scaler-ul (.pkl)
    # Folosim caile relative standard
    model = tf.keras.models.load_model('models/optimized_model.keras') # Atentie: am pus modelul optimizat
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("✅ Model AI și Preprocesor încărcate cu succes!\n")
except Exception as e:
    print(f"❌ EROARE CRITICA: Nu găsesc fișierele în folderul 'models/'. \nDetalii: {e}")
    print("Sfat: Ruleaza mai intai antrenarea retelei!")
    input("Apasa ENTER pentru a iesi...")
    sys.exit()

def get_user_input():
    """
    Functie care preia datele de la tastatura si le pune intr-un DataFrame.
    Formatul trebuie sa fie IDENTIC cu cel folosit la antrenare.
    """
    print("-" * 40)
    print("INTRODUCEȚI DETALIILE COMENZII:")
    try:
        # Preluam datele una cate una
        distanta = float(input("1. Distanța (km) [ex: 5.5]: "))
        
        print("   (1=Liber ... 5=Foarte Aglomerat)")
        trafic = int(input("2. Nivel Trafic (1-5): "))
        
        ora = float(input("3. Ora comenzii [ex: 14.5 pentru 14:30]: "))
        
        print("   (Luni, Marți, Miercuri, Joi, Vineri, Sâmbătă, Duminică)")
        ziua = input("4. Ziua săptămânii: ").capitalize() # .capitalize() repara daca userul scrie cu litera mica
        
        print("   (Bicicletă, Scuter, Dubiță, Camion)")
        vehicul = input("5. Tip Vehicul: ").capitalize()
        
        grad = float(input("6. Grad încărcare (0.1 - 1.0) [ex: 0.5]: "))
        
        # Returnam un DataFrame (tabel cu o singura linie)
        return pd.DataFrame({
            'Distanța (km)': [distanta],
            'Nivel trafic': [trafic],
            'Ora livrării': [ora],
            'Ziua săptămânii': [ziua],
            'Tip vehicul': [vehicul],
            'Grad de încărcare': [grad]
        })
    except ValueError:
        print("❌ Date introduse greșit! Te rog introdu numere unde este cazul.")
        return None

# --- BUCLA PRINCIPALA ---
while True:
    # 1. Cerem datele
    df_input = get_user_input()
    
    if df_input is not None:
        try:
            # 2. Preprocesare (Scalare + OneHotEncoding)
            # Folosim exact acelasi scaler ca la antrenare
            X_input = preprocessor.transform(df_input)
            
            # 3. Predictie
            # verbose=0 ascunde bara de progres
            predictie_min = model.predict(X_input, verbose=0)[0][0]
            
            # 4. Afisare Rezultat
            print(f"\n⏱️  TIMP ESTIMAT DE LIVRARE: {predictie_min:.0f} minute")
            print("-" * 40)
            
        except Exception as e:
            print(f"⚠️ Eroare la procesare: {e}")
            print("Verifica daca ai scris corect Ziua sau Tipul Vehiculului (exact ca in optiuni).")
            
    # Intrebam daca mai vrea o tura
    continuare = input("\nAltă simulare? (d/n): ")
    if continuare.lower() != 'd':
        print("La revedere! 👋")
        break