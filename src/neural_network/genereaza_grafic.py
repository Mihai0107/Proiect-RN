import matplotlib.pyplot as plt
# ... (Aici se presupune ca avem deja modelul definit si compilat anterior) ...

# Pornim antrenarea si salvam istoricul in variabila 'history' pentru a putea face graficele
history = model.fit(X, y, epochs=50, validation_split=0.2)

# Generam graficul pentru a vizualiza curba de invatare (sa vedem daca avem Overfitting)
plt.plot(history.history['loss'], label='Eroare Antrenare')
plt.plot(history.history['val_loss'], label='Eroare Validare')
plt.title('Progresul Învățării Modelului')
plt.ylabel('Eroare (MAE)')
plt.xlabel('Epoci')
plt.legend()
plt.savefig('grafic_invatare.png')
print("Grafic salvat!")