1. Arhitectura Rețelei Neuronale
Am proiectat și implementat o rețea neuronală de tip Feed-Forward (MLP) folosind biblioteca TensorFlow/Keras.

Structura Modelului:
Strat Intrare: Primește vectorul de date standardizat (Distanță, Trafic, etc.).

Strat Ascuns 1: 64 neuroni, funcție de activare ReLU. Rol: Extragerea caracteristicilor neliniare complexe.

Strat Ascuns 2: 32 neuroni, funcție de activare ReLU. Rol: Rafinarea informației înainte de predicție.

Strat Ieșire: 1 neuron, activare Lineară. Rol: Predicția valorii continue (durata în minute).

2. Antrenarea Modelului
Configurație:
Optimizator: Adam (Learning Rate adaptiv).

Funcție de Cost (Loss): MSE (Mean Squared Error) – pentru minimizarea pătratului erorilor.

Metrică: MAE (Mean Absolute Error) – pentru interpretarea ușoară a erorii în minute.

Mecanism de Siguranță: EarlyStopping (oprește antrenarea dacă val_loss nu scade timp de 10 epoci).

Rezultate Obținute:
Antrenarea s-a oprit automat pentru a preveni overfitting-ul.

Eroarea Medie (MAE) pe Train: ~ 6.3 minute.

Eroarea Medie (MAE) pe Test (Date noi): 6.45 minute.

Concluzie: Modelul este capabil să prezică durata unei livrări cu o abatere medie de doar 6.45 minute față de realitate, un rezultat excelent având în vedere variabilitatea traficului introdusă în date.

3. Sistemul de Predicție "Live Demo"
Proiectul include un script interactiv (demo_live.py) care permite testarea modelului în timp real.

Cum funcționează:
1. Scriptul încarcă modelul salvat (models/model_livrare.keras).

2. Încarcă scaler-ul și encoder-ul salvate (models/preprocessor.pkl) pentru a transforma datele introduse de utilizator exact ca la antrenare.

3. Utilizatorul introduce datele de la tastatură (Distanță, Trafic, Oră, Vehicul, etc.).

4. Rețeaua returnează predicția instantaneu.

Ghid de Utilizare:
Pentru a rula demo-ul, executați în terminal:python demo_live.py

4. Structura Finală a Fișierelor

project/
├── data/
│   ├── raw/                 # Date brute (CSV)
│   ├── train/               # Date antrenare (.npy)
│   ├── validation/          # Date validare (.npy)
│   └── test/                # Date test (.npy)
├── models/
│   ├── model_livrare.keras  # Rețeaua Neuronală antrenată
│   └── preprocessor.pkl     # Scaler-ul salvat
├── src/
│   ├── generare_date.py     # Script generare dataset
│   ├── analiza_eda.py       # Script analiză statistică
│   ├── preprocesare_finala.py # Script curățare + salvare preprocesor
│   ├── antrenare_retea.py   # Script antrenare model
│   └── demo_live.py         # Script interactiv de testare
├── grafic_antrenare.png     # Vizualizarea curbei de învățare
└── README.md                # Această documentație

5. Tehnologii Utilizate
Limbaj: Python 3.13

Data Science: Pandas, NumPy, Scikit-learn

Deep Learning: TensorFlow, Keras

Vizualizare: Matplotlib

Serializare: Joblib (pentru salvarea pipeline-ului de date)