# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Data:** [Data]  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** [Descriere sursă date - ex: senzori robot, dataset public, simulare]
* **Modul de achiziție:** ☐ Senzori reali / ☐ Simulare / ☐ Fișier extern / ☐ Generare programatică
* **Perioada / condițiile colectării:** [Ex: Noiembrie 2024 - Ianuarie 2025, condiții experimentale specifice]

### 2.2 Caracteristicile dataset-ului
* **Număr total de observații:** 5000 (inițial), redus la ~4950 după eliminarea outlierilor.
* **Număr de caracteristici (features):** 6 (Intrări) + 1 (Ieșire).
* **Tipuri de date:** ☒ Numerice / ☒ Categoriale / ☒ Temporale
* **Format fișiere:** ☒ CSV

### 2.3 Descrierea fiecărei caracteristici
| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| :--- | :--- | :--- | :--- | :--- |
| **Distanța** | numeric | km | Distanța totală de livrare | 0.5 – 150.0 |
| **Nivel trafic** | numeric | scală | Intensitatea traficului | 1 (Liber) – 5 (Aglomerat) |
| **Ora livrării** | numeric | oră | Ora la care a plecat curierul | 7.00 – 22.00 |
| **Ziua săptămânii** | categorial | - | Ziua calendaristică | Luni – Duminică |
| **Tip vehicul** | categorial | - | Vehiculul folosit | Bicicletă, Scuter, Dubiță, Camion |
| **Grad de încărcare**| numeric | % | Cât de plin e vehiculul | 0.1 (10%) – 1.0 (100%) |
| **Durata estimată** | numeric | min | **Variabila Țintă (Output)** | 5 – 706 min |
**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate
* **Media Distanței:** 25.21 km.
* **Media Duratei:** 105.63 minute (deviație standard mare, indicând variații semnificative).
* **Corelații:** S-a observat o corelație pozitivă puternică (0.60) între **Distanță** și **Durată**, confirmând că distanța este principalul predictor.

### 3.2 Analiza calității datelor
* **Valori lipsă:** S-au identificat **250** valori lipsă pe coloana `Nivel trafic` (reprezentând exact 5% din date).
* **Outlieri:** S-au identificat valori extreme la `Distanța` (ex: 150 km), mult peste media obișnuită.

### 3.3 Probleme identificate
1.  Lipsa datelor de trafic pentru 5% din comenzi (necesită imputare).
2.  Distanțe extreme care pot distorsiona modelul (necesită filtrare).
3.  Variabile categoriale (`Tip vehicul`, `Ziua`) care necesită transformare numerică (Encoding).

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor
* **Tratarea valorilor lipsă:** Valorile lipsă din `Nivel trafic` au fost completate cu **mediana** (valoare = 3.0), metodă robustă la outlieri.
* **Tratarea outlierilor:** S-au eliminat livrările cu distanțe care depășesc percentila 99, considerându-le erori sau cazuri excepționale irelevante pentru modelul general.

### 4.2 Transformarea caracteristicilor
* **Encoding:** S-a aplicat **One-Hot Encoding** pe `Ziua săptămânii` și `Tip vehicul` pentru a le transforma în vectori binari. 
* **Scalare:** S-a aplicat **StandardScaler** (Z-score) pe toate variabilele numerice (`Distanța`, `Trafic`, `Ora`, `Încărcare`) pentru a aduce datele la o medie de 0 și deviație de 1, esențial pentru convergența Rețelei Neuronale.

### 4.3 Structurarea seturilor de date
Datele au fost amestecate și împărțite astfel:
* **70% Train** (~3500 exemple) – pentru antrenare.
* **15% Validation** (~730 exemple) – pentru optimizare hiperparametri.
* **15% Test** (~730 exemple) – pentru evaluarea finală (date nevăzute).
**Notă:** Scalarea s-a "învățat" doar pe setul Train pentru a evita Data Leakage.

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---
