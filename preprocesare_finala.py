import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib # <--- NOU
import os


df = pd.read_csv('data/raw/delivery_data.csv')
print("1. Date încărcate.")


mediana_trafic = df['Nivel trafic'].median()
df['Nivel trafic'] = df['Nivel trafic'].fillna(mediana_trafic)

limita_superioara = df['Distanța (km)'].quantile(0.99)
df = df[df['Distanța (km)'] <= limita_superioara]


X = df.drop('Durata estimată (min)', axis=1)
y = df['Durata estimată (min)']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)


numeric_features = ['Distanța (km)', 'Nivel trafic', 'Ora livrării', 'Grad de încărcare']
categorical_features = ['Ziua săptămânii', 'Tip vehicul']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


X_train_processed = preprocessor.fit_transform(X_train)


X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()


os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('models', exist_ok=True)

np.save('data/train/X_train.npy', X_train_processed)
np.save('data/train/y_train.npy', y_train)
np.save('data/validation/X_val.npy', X_val_processed)
np.save('data/validation/y_val.npy', y_val)
np.save('data/test/X_test.npy', X_test_processed)
np.save('data/test/y_test.npy', y_test)


joblib.dump(preprocessor, 'models/preprocessor.pkl') 

print("✅ Datele și Preprocesorul au fost salvate!")