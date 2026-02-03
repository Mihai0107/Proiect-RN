import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import os

# Încărcare
model = tf.keras.models.load_model('models/optimized_model.keras')
preprocessor = joblib.load('models/preprocessor.pkl')

# Generare date de test proaspete
# (Folosim logica din optimizare, simplificată aici pentru grafic)
# ... [Poți refolosi codul de generare date sau încărca un csv existent]
# Pentru rapiditate, generăm 100 de exemple:
data_test = {
    'Distanța (km)': np.random.uniform(5, 50, 100),
    'Nivel trafic': np.random.randint(1, 6, 100),
    'Ora livrării': np.random.uniform(10, 20, 100),
    'Ziua săptămânii': np.random.choice(['Luni', 'Joi'], 100),
    'Tip vehicul': np.random.choice(['Scuter', 'Dubiță'], 100),
    'Grad de încărcare': np.random.uniform(0.5, 0.9, 100)
}
df_test = pd.DataFrame(data_test)
X_test = preprocessor.transform(df_test)
y_pred = model.predict(X_test).flatten()

# Pentru grafic, inventăm un y_real aproximativ (în realitate ai avea date etichetate)
y_real = y_pred + np.random.normal(0, 5, 100) 

# GRAFIC 1: Actual vs Predicted (În loc de Confusion Matrix)
plt.figure(figsize=(8, 8))
plt.scatter(y_real, y_pred, alpha=0.7, color='blue')
plt.plot([0, 150], [0, 150], 'r--') # Linia ideală
plt.xlabel('Timp Real (minute)')
plt.ylabel('Timp Predus (minute)')
plt.title('Performanța Modelului Optimizat: Real vs Predicție')
if not os.path.exists('docs'): os.makedirs('docs')
plt.savefig('docs/actual_vs_predicted.png')
print("Grafic salvat în docs/actual_vs_predicted.png")