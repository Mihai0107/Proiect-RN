import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n🚀 PORNIRE SISTEM DE PREDICȚIE LIVE...")


try:
    model = tf.keras.models.load_model('models/model_livrare.keras')
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("✅ Model și Preprocesor încărcate cu succes!\n")
except:
    print("❌ EROARE: Nu găsesc fișierele în folderul 'models/'. Rulează pașii anteriori.")
    exit()

def get_user_input():
    print("-" * 40)
    print("INTRODUCEȚI DETALIILE COMENZII:")
    try:
        distanta = float(input("1. Distanța (km) [ex: 5.5]: "))
        
        print("   (1=Liber ... 5=Foarte Aglomerat)")
        trafic = int(input("2. Nivel Trafic (1-5): "))
        
        ora = float(input("3. Ora comenzii [ex: 14.5 pentru 14:30]: "))
        
        print("   (Luni, Marți, Miercuri, Joi, Vineri, Sâmbătă, Duminică)")
        ziua = input("4. Ziua săptămânii: ").capitalize()
        
        print("   (Bicicletă, Scuter, Dubiță, Camion)")
        vehicul = input("5. Tip Vehicul: ").capitalize()
        
        grad = float(input("6. Grad încărcare (0.1 - 1.0) [ex: 0.5]: "))
        
        return pd.DataFrame({
            'Distanța (km)': [distanta],
            'Nivel trafic': [trafic],
            'Ora livrării': [ora],
            'Ziua săptămânii': [ziua],
            'Tip vehicul': [vehicul],
            'Grad de încărcare': [grad]
        })
    except ValueError:
        print("❌ Date introduse greșit! Încearcă din nou.")
        return None


while True:
    df_input = get_user_input()
    
    if df_input is not None:

        try:
            X_input = preprocessor.transform(df_input)
            

            predictie_min = model.predict(X_input, verbose=0)[0][0]
            
            print(f"\n⏱️  TIMP ESTIMAT DE LIVRARE: {predictie_min:.0f} minute")
            print("-" * 40)
        except Exception as e:
            print(f"Eroare la procesare (probabil ai scris greșit ziua/vehiculul): {e}")
            
    continuare = input("\nAltă simulare? (d/n): ")
    if continuare.lower() != 'd':
        print("La revedere! 👋")
        break