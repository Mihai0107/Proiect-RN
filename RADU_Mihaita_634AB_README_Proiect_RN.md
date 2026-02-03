## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Radu Mihaita-Gheorghe |
| **Grupa / Specializare** | 634AB |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | [https://github.com/Mihai0107/Proiect-RN.git] |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow/Keras, CustomTkinter, Scikit-Learn) |
| **Domeniul Industrial de Interes (DII)** | Logistică / Last Mile Delivery |
| **Tip Rețea Neuronală** | MLP (Multi-Layer Perceptron) - Regresie |

### Rezultate Cheie (Versiunea Finală vs Etapa 5 Baseline)

*Notă: Fiind o problemă de regresie, Accuracy este echivalată cu R² Score, iar F1 este înlocuit cu MAE.*

| Metric | Țintă Minimă | Rezultat Etapa 5 (Baseline) | Rezultat Final (Etapa 6) | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (R² Score) | ≥70% (0.70) | ~0.72 | **0.91** | +19% | [✓] |
| MAE (Eroare Medie) | ≤ 3.0 min | 3.85 min | **2.15 min** | -1.7 min | [✓] |
| Latență Inferență | < 100 ms | 48 ms | **35 ms** | -13 ms | [✓] |
| Contribuție Date Originale | ≥40% | 100% | **100%** | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | - | **4** | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Radu Mihaita-Gheorghe , declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Sectorul de logistică și "Last Mile Delivery" se confruntă cu o problemă majoră: incertitudinea timpului de sosire. Estimările clasice (bazate doar pe distanță / viteză medie) eșuează adesea în mediul urban aglomerat, ducând la întârzieri și clienți nemulțumiți. Companiile pierd resurse valoroase planificând rute ineficiente bazate pe estimări statice ("Ajungem în 30 min" vs realitate 50 min).

Proiectul propune un Sistem cu Inteligență Artificială (SIA) care prezice durata livrării luând în calcul variabile complexe simultan: distanța, nivelul de trafic (1-5), ora comenzii, ziua săptămânii și tipul vehiculului folosit.

### 2.2 Beneficii Măsurabile Urmărite

1. **Predicție precisă:** Reducerea erorii medii de estimare sub 3 minute.
2. **Optimizare operațională:** Capacitatea de a procesa o cerere în sub 50ms pentru integrare în timp real.
3. **Adaptabilitate:** Modelul ține cont de factori variabili (trafic, oră de vârf) invizibili pentru calculatoarele clasice de distanță.
4. **Satisfacția clienților:** Oferirea unui interval de timp (ETA) realist, reducând reclamațiile.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Estimare realistă în trafic | Regresie non-liniară multi-variabilă | Neural Network (Keras) | MAE < 3 min |
| Interfață accesibilă curierilor | Aplicație Desktop intuitivă | UI (CustomTkinter) | Usability (Click-to-Predict) |
| Calcul rapid pentru dispecerat | Model optimizat (.h5) | Inference Engine | Latency < 50ms |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Generare proprie (Sintetice) bazate pe logică reală |
| **Sursa concretă** | Script Python `src/data_acquisition/generare_date.py` |
| **Număr total observații finale (N)** | 6000 |
| **Număr features** | 6 (Distanță, Trafic, Oră, Zi, Vehicul, Încărcare) |
| **Tipuri de date** | Numerice (3) + Categoriale (3) |
| **Format fișiere** | CSV |
| **Perioada colectării/generării** | Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 6000 |
| **Observații originale (M)** | 6000 |
| **Procent contribuție originală** | **100%** |
| **Tip contribuție** | Generare algoritmică bazată pe logica de trafic real |
| **Locație cod generare** | `src/data_acquisition/generare_date.py` |
| **Locație date originale** | `data/` |

**Descriere metodă generare/achiziție:**

Datele au fost generate folosind un script Python care simulează condiții reale de trafic urban. Nu s-au folosit date random simple, ci o logică de calcul complexă care include: penalizări dinamice pentru ore de vârf (08:00-10:00, 16:00-19:00), multiplicatori exponențiali pentru trafic (Nivel 1-5), factori de viteză specifici vehiculului (ex: bicicleta e mai lentă dar mai puțin afectată de trafic) și zgomot gaussian pentru a simula imprevizibilul.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 4200 |
| Validation | 15% | 900 |
| Test | 15% | 900 |

**Preprocesări aplicate:**
- **StandardScaler:** Pe variabilele numerice (Distanță, Trafic, Oră) pentru a aduce media la 0 și deviația la 1.
- **OneHotEncoder:** Pentru variabilele categoriale (Ziua, Vehicul).
- **Pipeline:** Integrare `ColumnTransformer` salvat ca `.pkl` pentru consistență la inferență.

**Referințe fișiere:** `data/delivery_data.csv`, `models/preprocessor.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging** | Python (Pandas/Numpy) | Generare date sintetice realiste și export CSV | `src/data_acquisition/` |
| **Neural Network** | TensorFlow/Keras | Antrenare model regresie și optimizare | `src/neural_network/` |
| **UI Application** | CustomTkinter | Interfață grafică modernă pentru predicție | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/Diagrama.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare input utilizator | Start aplicație | Buton "Calculează" apăsat |
| `VALIDATE_INPUT` | Verificare tip date (numeric) | Input primit | Date valide / Eroare |
| `PREPROCESS` | Scalare și Encoding | Date valide | Input formatat pentru RN |
| `INFERENCE` | Predicție model .keras | Input formatat | Timp brut estimat |
| `POSTPROCESS` | Adăugare logică business | Timp brut | Rezultat final |
| `DISPLAY` | Afișare rezultat în GUI | Rezultat final | Buton "Reset" apăsat |

**Justificare alegere arhitectură State Machine:**
Structura secvențială asigură că nu se face inferență pe date invalide (ceea ce ar crăpa aplicația) și separă logica de afișare de cea de calcul, permițând actualizarea modelului în backend fără a schimba interfața grafică.

### 4.3 Actualizări State Machine în Etapa 6

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| Reset State | Absent | Prezent (Buton Reset) | Permite predicții multiple rapid |
| Validation State | Implicit | Try-Except Blocks | Previne crash la input non-numeric |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input Layer (Features preprocesate) → Dense(128, activation='relu') → Dense(64, activation='relu') → Dense(32, activation='relu') → Output Layer(1) [Linear activation - implicit]

**Justificare alegere arhitectură:**
Am ales o arhitectură **Feed-Forward (MLP)** cu 3 straturi ascunse descrescătoare. Problema fiind una de regresie pe date tabulare, CNN sau RNN nu erau necesare. Adâncimea (3 straturi) permite captarea relațiilor non-liniare complexe (ex: interacțiunea dintre Trafic 5 și Ora de Vârf), iar `ReLU` asigură eficiență computațională.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Valoare standard Adam, convergență stabilă |
| Batch Size | 32 | Generalizare mai bună decât batch-uri mari (128) |
| Epochs | 50 | Early stopping activat pentru a preveni overfitting |
| Optimizer | Adam | Standardul industriei pentru regresie tabulară |
| Loss Function | MAE | Robust la outlieri, ușor de interpretat (minute) |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | MAE (Eroare) | R² Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | [64, 32], Batch=32 | 3.85 min | 0.72 | Rapid | Referință acceptabilă |
| Exp 1 | Batch size 128 | 4.10 min | 0.69 | Foarte Rapid | Batch mare a redus generalizarea |
| Exp 2 | **[128, 64, 32], Batch=32** | **2.15 min** | **0.91** | **Mediu** | **BEST MODEL - Arhitectură mai adâncă** |
| Exp 3 | Learning Rate 0.0001 | 4.50 min | 0.65 | Lent | Convergență incompletă |
| **FINAL** | **Exp 2** | **2.15 min** | **0.91** | **Mediu** | **Modelul folosit în producție** |

**Justificare alegere model final:**
Arhitectura din Experimentul 2 a oferit cea mai mică eroare medie (2.15 min). Adăugarea unui strat suplimentar de 128 neuroni a permis rețelei să învețe mai bine nuanțele traficului extrem, lucru pe care modelul baseline îl subestima.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.keras`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy (R² Score equivalent)** | **91%** | ≥70% | [✓] |
| **MAE (Eroare Medie)** | **2.15 min** | ≤ 3.0 min | [✓] |
| **MSE (Mean Squared Error)** | **8.5** | ≤ 10.0 | [✓] |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| MAE | 3.85 min | 2.15 min | -1.7 min (Eroare redusă cu 44%) |
| R² Score | 0.72 | 0.91 | +19% |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix (Adaptat: Scatter Plot)

**Locație:** `docs/grafic_antrenare.png` (și Actual vs Predicted)

*Notă: Matricea de confuzie se aplică clasificării. Pentru regresie, utilizăm graficul Actual vs Predicted.*

**Interpretare:**
Punctele sunt grupate strâns în jurul liniei ideale (diagonala roșie), indicând o precizie ridicată. Nu există bias-uri majore (nu supraestimează sau subestimează sistematic întregul set de date), deși la valori extreme (distanțe foarte mari) varianța crește ușor.

### 6.3 Analiza Top 5 Erori (Date Reale din Testare)

| # | Input (descriere scurtă) | Predicție RN | Timp Real | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| #216 | Bicicletă, 11km, Ora 9.9 | 46.3 min | 70.6 min | Subestimare impact trafic dimineață pe bicicletă. | Curierul întârzie 24 min → reclamație client. |
| #120 | Bicicletă, 54km, Ora 16.0 | 200.2 min | 223.8 min | Distanță extremă + oră vârf. Date rare în training. | Estimare optimistă pe curse lungi. |
| #217 | Dubiță, 20km, Trafic 3 | 63.3 min | 82.7 min | Subestimare impact trafic urban pe vehicul mare. | Planificare flotă eronată. |
| #94 | Dubiță, 36km, Trafic 5 | 115.7 min | 97.8 min | Supraestimare la trafic maxim (penalizare prea dură). | Timp tampon inutil alocat. |
| #498 | Bicicletă, 25km, Ora 15.9 | 91.7 min | 74.4 min | Penalizare prematură pentru ora de vârf (ora 16). | Alocare ineficientă a resurselor. |

### 6.4 Validare în Context Industrial

Ce înseamnă rezultatele pentru aplicația reală:

Modelul are o eroare medie de +/- 2 minute, ceea ce este excelent pentru livrări urbane, unde o marjă de eroare de câteva minute este insesizabilă pentru client. Erorile mari apar doar în cazuri extreme (distanțe >50km cu bicicleta), care sunt statistice rare (outlieri) și nu afectează majoritatea operațiunilor zilnice.

Pragul de acceptabilitate pentru domeniu: Eroare medie (MAE) ≤ 5-10 minute (toleranța standard a clientului). Status: Atins (2.15 min < 5 min). Plan de îmbunătățire: Colectarea de date reale suplimentare pentru rutele lungi (>50km) pentru a reduce erorile extreme.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `model_livrare.keras` | `optimized_model.keras` | Reducere eroare MAE cu 1.7 min |
| **Interfață** | Fereastră fixă | Scrollable Frame + Resizable | Compatibilitate cu ecrane mici |
| **Funcționalitate** | Calcul unic | Buton Resetare Câmpuri | Usability crescut pentru dispeceri |
| **Deployment** | Script Python | Executabil (.exe) | Livrare profesională fără deps Python |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

Screenshot-ul prezintă aplicația în modul Dark Mode, cu toate câmpurile completate și rezultatul predicției (Verde) afișat clar utilizatorului.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `dist/aplicatie.exe` (Executabil funcțional)

**Fluxul demonstrat:**
1.  **Input:** Utilizatorul introduce distanța (12.5 km), Trafic (3), Vehicul (Scuter).
2.  **Procesare:** Aplicația preia datele, le scalează folosind `preprocessor.pkl`.
3.  **Inferență:** Modelul `optimized_model.keras` prezice timpul.
4.  **Decizie:** Rezultatul este afișat instantaneu pe ecran (verde).

**Latență măsurată end-to-end:** 35 ms

---

## 8. Structura Repository-ului Final

proiect-rn-mihai0107/

│

├── Mihaita_Gheorghe_Radu_634AB_README_Proiect_RN.md  # ← ACEST FIȘIER

│

├── docs/

│   ├── actual_vs_predicted.png             # Grafic performanță (Scatter)

│   ├── grafic_antrenare.png                # Curba de învățare

│   ├── Diagrama.png                        # State Machine

│   └── screenshots/

│       └── inference_optimized.png         # Screenshot UI Final

│

├── data/

│   └── livrari_dataset.csv                 # Dataset generat

│

├── dist/                                   # Executabil

│   └── aplicatie.exe

│

├── src/

│   ├── data_acquisition/                   # Generare date

│   │   └── generare_date.py

│   │

│   ├── neural_network/                     # Model RN

│   │   ├── antrenare_retea.py

│   │   ├── optimizare.py

│   │   ├── antrenare_bonus.py

│   │   ├── analiza_erori.py

│   │   └── genereaza_grafic.py

│   │

│   ├── preprocessing/                      # Preprocesare

│   │   ├── preprocesare_finala.py

│   │   └── analiza_eda.py

│   │

│   └── app/                                # UI Application

│       ├── aplicatie.py

│       └── demo_live.py

│

├── models/

│   ├── optimized_model.keras               # Model FINAL optimizat

│   └── preprocessor.pkl                    # Scaler date

│

├── aplicatie.spec                          # Configurare PyInstaller

├── requirements.txt                        # Dependențe Python

└── .gitignore

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |



### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - R2=0.72, MAE=3.85 (Baseline)" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - R2=0.91, MAE=2.15 (Optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare
```text
Python >= 3.10
pip >= 21.0
OS: Windows 10/11 (Recomandat pentru CustomTkinter)

### 9.2 Instalare
# 1. Clonare repository
git clone [https://github.com/Mihai0107/Proiect-RN.git](https://github.com/Mihai0107/Proiect-RN.git)
cd Proiect-RN

# 2. Creare mediu virtual (recomandat)
python -m venv venv
# Activare Windows:
venv\Scripts\activate
# Activare Linux/Mac:
source venv/bin/activate

# 3. Instalare dependențe
pip install -r requirements.txt

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Generare și Preprocesare Date (dacă rulați de la zero)
# Generează datele sintetice
python src/data_acquisition/generare_date.py

# Curăță datele, face split train/test și salvează scaler-ul
python src/preprocessing/preprocesare_finala.py

# Pasul 2: Antrenare Model (pentru reproducere rezultate)
# Antrenează modelul și salvează optimized_model.keras
python src/neural_network/antrenare_retea.py

# (Opțional) Rulare optimizare hiperparametri
python src/neural_network/optimizare.py

# Pasul 3: Evaluare Model și Generare Grafice
# Calculează metricile finale și generează graficele de analiză
python src/neural_network/analiza_finala.py
python src/neural_network/genereaza_grafic.py

# Pasul 4: Lansare Aplicație UI
# Pornește interfața grafică CustomTkinter
python src/app/aplicatie.py

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

Obiectiv Definit (Secțiunea 2)	Target	Realizat	Status
Predicție precisă (MAE)	< 3 min	2.15 min	[✓]
Timp răspuns	< 100ms	35ms	[✓]
Accuracy (echiv. R2)	≥70%	91%	[✓]

### 10.2 Ce NU Funcționează – Limitări Cunoscute

Date Sintetice: Deși logica e complexă, datele nu surprind accidente reale sau blocaje spontane de trafic.

Input Subiectiv: Nivelul de trafic (1-5) este introdus de utilizator, ceea ce poate introduce erori umane dacă utilizatorul nu estimează corect.

Cazuri Extreme: Modelul tinde să subestimeze timpul pentru curse foarte lungi (>50km) făcute cu vehicule lente (biciclete), din lipsa de exemple suficiente la antrenare.

### 10.3 Lecții Învățate (Top 5)

Preprocesarea e cheia: Normalizarea datelor a avut un impact mai mare asupra convergenței decât numărul de neuroni.

Deployment-ul este complex: Transformarea în .exe necesită gestionarea atentă a căilor relative și a importurilor ascunse (sklearn).

Arhitectura contează: Trecerea de la 2 la 3 straturi a redus eroarea semnificativ (captând non-linearități).

UX: Un buton de Reset pare trivial, dar schimbă complet fluxul de lucru al utilizatorului.

Early Stopping: Esențial pentru a preveni overfitting-ul și a economisi timp de antrenare.

### 10.4 Retrospectivă

Dacă aș reîncepe proiectul, aș investi mai mult timp în colectarea unor date reale (folosind un API de Google Maps) în loc să mă bazez pe date sintetice, pentru a captura mai bine variabilitatea reală a traficului. De asemenea, aș implementa un sistem automat de tuning de hiperparametri (Keras Tuner) de la început, în loc să testez manual configurațiile.

### 10.5 Direcții de Dezvoltare Ulterioară

Termen,Îmbunătățire Propusă,Beneficiu Estimat
Short-term,Integrare API Meteo,+5% precizie pe timp de ploaie
Medium-term,Preluare automată Trafic (Google API),Eliminare eroare umană la input
Long-term,Aplicație Mobilă,Accesibilitate pentru curieri pe teren

## 11. Bibliografie

Li, Y., Fu, Z., 2022. Travel time prediction based on deep learning: A survey. IEEE Access. 10, 12268-12285. https://doi.org/10.1109/ACCESS.2022.3146743

TensorFlow Documentation: Regression with Keras. https://www.tensorflow.org/tutorials/keras/regression

Scikit-Learn Documentation: Preprocessing data. https://scikit-learn.org/stable/modules/preprocessing.html

CustomTkinter Documentation. https://github.com/TomSchimansky/CustomTkint

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy (R²) ≥70%** pe test set (verificat în `results/final_metrics.json` - rezultat 0.91)
- [x] **MAE ≤ 3.0** (echivalent F1 performant pentru regresie - rezultat 2.15 min)
- [x] **Contribuție 100% date originale** (verificabil în `data/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [x] **Confusion matrix (Scatter Plot)** generată și interpretată (Secțiunea 6.2)
- [x] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [x] **Demonstrație end-to-end** disponibilă (Executabil funcțional)

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate cu date reale)
- [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă cu Secțiunea 8
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [x] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [x] **Tag `v0.6-optimized-final`** creat și pushed
- [x] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [x] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

Verificare Anti-Plagiat

[x] Model antrenat de la zero

[x] Minimum 40% date originale

[x] Cod propriu sau clar atribuit

##Note Finale
Versiune document: FINAL pentru examen

Ultima actualizare: 03.02.2026

Tag Git: v0.6-optimized-final

