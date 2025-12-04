import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os


print("1. Încărcare date...")
try:
    X_train = np.load('data/train/X_train.npy')
    y_train = np.load('data/train/y_train.npy')
    X_val = np.load('data/validation/X_val.npy')
    y_val = np.load('data/validation/y_val.npy')
    X_test = np.load('data/test/X_test.npy')
    y_test = np.load('data/test/y_test.npy')
except FileNotFoundError:
    print("EROARE: Rulează mai întâi preprocesare_finala.py!")
    exit()


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


print("2. Antrenare...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)


test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ MAE Final pe Test: {test_mae:.2f} minute")


os.makedirs('models', exist_ok=True)
model.save('models/model_livrare.keras')
print("💾 Modelul a fost salvat în 'models/model_livrare.keras'")