import matplotlib.pyplot as plt
# ... (codul tau de antrenare model) ...

history = model.fit(X, y, epochs=50, validation_split=0.2)

# Desenare grafic
plt.plot(history.history['loss'], label='Eroare Antrenare')
plt.plot(history.history['val_loss'], label='Eroare Validare')
plt.title('Progresul Învățării Modelului')
plt.ylabel('Eroare (MAE)')
plt.xlabel('Epoci')
plt.legend()
plt.savefig('grafic_invatare.png')
print("Grafic salvat!")