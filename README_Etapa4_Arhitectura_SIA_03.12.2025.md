# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [Adaugă Link-ul Tău Aici]
**Data:** [Data Curentă]  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este definit și compilat.**

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Estimarea precisă a timpului de livrare pentru logistică | Regresie neuronală (RN) antrenată pe date istorice simulate, cu eroare medie (MAE) ~6.45 minute | `src/antrenare_retea.py` + `models/model_livrare.keras` |
| Răspuns rapid pentru dispeceri la introducerea unei comenzi | Inferență în timp real (< 1 secundă) pe baza datelor de intrare (distanță, trafic, vehicul) | `src/demo_live.py` (User Interface) + `models/preprocessor.pkl` |
| Generarea de scenarii diverse pentru antrenarea curierilor | Simulator de date capabil să genereze mii de situații (trafic intens, distanțe mari) | `src/generare_date.py` (Data Acquisition Module) |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

### Contribuția originală la setul de date:

**Total observații finale:** 5000 (Etapa 3 + Etapa 4)
**Observații originale:** 5000 (100%)

**Tipul contribuției:**
[x] Date generate prin simulare fizică / logică de business
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Am generat un set de date sintetic complet (5000 de instanțe) care simulează condițiile reale de livrare urbană. Logica de simulare a inclus definirea unor viteze medii specifice per tip de vehicul (ex: Bicicletă=15km/h, Camion=40km/h) și aplicarea unor factori de penalizare bazați pe nivelul de trafic (de la 1.0x pentru trafic liber la 3.0x pentru trafic blocat).
De asemenea, am introdus un factor de "zgomot" gaussian (deviație standard 5 minute) și o influență a gradului de încărcare asupra vitezei, pentru a mima imprevizibilitatea din lumea reală.

**Locația codului:** `src/generare_date.py`
**Locația datelor:** `data/raw/delivery_data.csv`

**Dovezi:**
- Scriptul de generare care conține logica de business (`calculeaza_durata`).
- Fisierul CSV rezultat care conține toate coloanele specificate (Distanță, Trafic, Oră, etc.).

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Fluxul logic al aplicației (Prediction Workflow):**

[START] ↓ IDLE (Așteptare input utilizator - Stand-by) ↓ [Eveniment: Operatorul introduce date: Distanță, Trafic...] ACQUIRE_USER_INPUT ↓ VALIDATE_DATA (Verificare valori: Distanță > 0, Trafic 1-5) ├─ [Date Invalide] ──────────→ DISPLAY_ERROR_MESSAGE ──┐ │ │ └─ [Date Valide] │ ↓ │ LOAD_RESOURCES (Model .keras + Preprocesor .pkl) │ ↓ │ PREPROCESS_INPUT (Scalare + One-Hot Encoding) │ ↓ │ RN_INFERENCE (Predicție durată livrare - Model FeedForward)│ ↓ │ DISPLAY_RESULT (Afișare "Timp estimat: X minute") │ ↓ │ WAIT_NEXT_ACTION ─────────────────────────────────────────┘ ├─ [Altă simulare] ──→ IDLE └─ [Ieșire] ──→ STOP

**Legendă obligatorie:**

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **Event-Driven / On-Demand Prediction** (Predicție la cerere), deoarece proiectul răspunde nevoii punctuale a unui operator uman (dispecer) de a estima durata unei livrări specifice înainte de a o aloca. Acest flux diferă de sistemele de monitorizare continuă (ex: vibrații), fiind declanșat doar la acțiunea utilizatorului.

**Stările principale sunt:**
1.  **IDLE (Așteptare):** Sistemul este în stare de repaus, consumând resurse minime, așteptând ca operatorul să introducă datele prin interfața `demo_live.py`.
2.  **VALIDATE_DATA:** O stare critică de "Sanity Check". Aici se verifică integritatea datelor (ex: distanța să nu fie negativă, traficul să fie între 1 și 5). Această stare previne introducerea de date eronate (Garbage In) care ar duce la predicții invalide.
3.  **PREPROCESS_INPUT:** Etapa de transformare a datelor brute. Convertește textul (ex: "Joi", "Dubiță") în vectori numerici (One-Hot Encoding) și scalează valorile numerice (StandardScaler) folosind parametrii salvați în `preprocessor.pkl`.
4.  **RN_INFERENCE:** Execuția modelului de Rețea Neuronală (Feed-Forward). Această stare este computațional intensivă dar optimizată pentru a returna rezultatul în < 1 secundă.

**Tranzițiile critice sunt:**
- **[VALIDATE_DATA] → [DISPLAY_ERROR]:** Tranziție automată declanșată dacă validarea eșuează. Protejează integritatea sistemului și previne blocarea aplicației.
- **[PREPROCESS_INPUT] → [RN_INFERENCE]:** Tranziție condiționată de succesul încărcării resurselor (model + preprocesor).

---

### 4. Scheletul Complet al celor 3 Module Cerute

Toate cele 3 module pornesc și rulează fără erori.

#### **Modul 1: Data Logging / Acquisition**
* **Fișier:** `src/generare_date.py`
* **Funcționalitate:** Generează dataset-ul sintetic pe baza parametrilor fizici (viteză, trafic, penalizări).
* **Status:** [x] Funcțional, produce `data/raw/delivery_data.csv` cu 5000 observații.

#### **Modul 2: Neural Network Module**
* **Fișier:** `src/antrenare_retea.py`
* **Funcționalitate:** Definește arhitectura Keras (Dense 64 -> Dense 32 -> Dense 1), compilează modelul și îl antrenează. Salvează modelul în `models/model_livrare.keras`.
* **Status:** [x] Funcțional, modelul este definit, antrenat (MAE 6.45) și salvat.

#### **Modul 3: Web Service / UI**
* **Fișier:** `src/demo_live.py`
* **Funcționalitate:** Interfață consolă interactivă. Primește input de la utilizator, validează datele, apelează preprocesorul (`models/preprocessor.pkl`) și modelul pentru inferență.
* **Status:** [x] Funcțional, demonstrează pipeline-ul complet (Input -> Predicție).

---

## Structura Repository-ului la Finalul Etapei 4

project-name/ ├── data/ │ ├── raw/ # CSV generat (delivery_data.csv) │ ├── train/ # .npy files │ ├── validation/ # .npy files │ └── test/ # .npy files ├── models/ │ ├── model_livrare.keras # Modelul Keras salvat │ └── preprocessor.pkl # Scaler-ul salvat ├── src/ │ ├── generare_date.py # Modul 1: Data Acquisition │ ├── analiza_eda.py # (Din Etapa 3) │ ├── preprocesare_finala.py # (Din Etapa 3) │ ├── antrenare_retea.py # Modul 2: Neural Network │ └── demo_live.py # Modul 3: UI ├── docs/ │ ├── state_machine.png # Diagrama SM (vezi instrucțiunile) │ └── grafic_antrenare.png # Dovada antrenării ├── README.md └── README_Etapa4_Arhitectura_SIA_03.12.2025.md

Aceasta este versiunea **completă și finală** a fișierului `README_Etapa4_Arhitectura_SIA_03.12.2025.md`.

Am completat toate secțiunile (Tabel, Contribuție Date, Diagramă, Module, Checklist) bazându-mă strict pe codul și rezultatele pe care le-am generat împreună (MAE 6.45 min, 100% date generate, structura Python).

Poți da **Copy** la tot blocul de mai jos și **Paste** în fișierul tău.

```markdown
# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [Adaugă Link-ul Tău Aici]
**Data:** [Data Curentă]  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este definit și compilat.**

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Estimarea precisă a timpului de livrare pentru logistică | Regresie neuronală (RN) antrenată pe date istorice simulate, cu eroare medie (MAE) ~6.45 minute | `src/antrenare_retea.py` + `models/model_livrare.keras` |
| Răspuns rapid pentru dispeceri la introducerea unei comenzi | Inferență în timp real (< 1 secundă) pe baza datelor de intrare (distanță, trafic, vehicul) | `src/demo_live.py` (User Interface) + `models/preprocessor.pkl` |
| Generarea de scenarii diverse pentru antrenarea curierilor | Simulator de date capabil să genereze mii de situații (trafic intens, distanțe mari) | `src/generare_date.py` (Data Acquisition Module) |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

### Contribuția originală la setul de date:

**Total observații finale:** 5000 (Etapa 3 + Etapa 4)
**Observații originale:** 5000 (100%)

**Tipul contribuției:**
[x] Date generate prin simulare fizică / logică de business
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Am generat un set de date sintetic complet (5000 de instanțe) care simulează condițiile reale de livrare urbană. Logica de simulare a inclus definirea unor viteze medii specifice per tip de vehicul (ex: Bicicletă=15km/h, Camion=40km/h) și aplicarea unor factori de penalizare bazați pe nivelul de trafic (de la 1.0x pentru trafic liber la 3.0x pentru trafic blocat).
De asemenea, am introdus un factor de "zgomot" gaussian (deviație standard 5 minute) și o influență a gradului de încărcare asupra vitezei, pentru a mima imprevizibilitatea din lumea reală.

**Locația codului:** `src/generare_date.py`
**Locația datelor:** `data/raw/delivery_data.csv`

**Dovezi:**
- Scriptul de generare care conține logica de business (`calculeaza_durata`).
- Fisierul CSV rezultat care conține toate coloanele specificate (Distanță, Trafic, Oră, etc.).

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Fluxul logic al aplicației (Prediction Workflow):**

```

[START]
↓
IDLE (Așteptare input utilizator - Stand-by)
↓ [Eveniment: Operatorul introduce date: Distanță, Trafic...]
ACQUIRE\_USER\_INPUT
↓
VALIDATE\_DATA (Verificare valori: Distanță \> 0, Trafic 1-5)
├─ [Date Invalide] ──────────→ DISPLAY\_ERROR\_MESSAGE ──┐
│                                                      │
└─ [Date Valide]                                       │
↓                                                 │
LOAD\_RESOURCES (Model .keras + Preprocesor .pkl)          │
↓                                                 │
PREPROCESS\_INPUT (Scalare + One-Hot Encoding)             │
↓                                                 │
RN\_INFERENCE (Predicție durată livrare - Model FeedForward)│
↓                                                 │
DISPLAY\_RESULT (Afișare "Timp estimat: X minute")         │
↓                                                 │
WAIT\_NEXT\_ACTION ─────────────────────────────────────────┘
├─ [Altă simulare] ──→ IDLE
└─ [Ieșire] ──→ STOP

```

**Legendă obligatorie:**

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **Event-Driven / On-Demand Prediction** (Predicție la cerere), deoarece proiectul răspunde nevoii punctuale a unui operator uman (dispecer) de a estima durata unei livrări specifice înainte de a o aloca. Acest flux diferă de sistemele de monitorizare continuă (ex: vibrații), fiind declanșat doar la acțiunea utilizatorului.

**Stările principale sunt:**
1.  **IDLE (Așteptare):** Sistemul este în stare de repaus, consumând resurse minime, așteptând ca operatorul să introducă datele prin interfața `demo_live.py`.
2.  **VALIDATE_DATA:** O stare critică de "Sanity Check". Aici se verifică integritatea datelor (ex: distanța să nu fie negativă, traficul să fie între 1 și 5). Această stare previne introducerea de date eronate (Garbage In) care ar duce la predicții invalide.
3.  **PREPROCESS_INPUT:** Etapa de transformare a datelor brute. Convertește textul (ex: "Joi", "Dubiță") în vectori numerici (One-Hot Encoding) și scalează valorile numerice (StandardScaler) folosind parametrii salvați în `preprocessor.pkl`.
4.  **RN_INFERENCE:** Execuția modelului de Rețea Neuronală (Feed-Forward). Această stare este computațional intensivă dar optimizată pentru a returna rezultatul în < 1 secundă.

**Tranzițiile critice sunt:**
- **[VALIDATE_DATA] → [DISPLAY_ERROR]:** Tranziție automată declanșată dacă validarea eșuează. Protejează integritatea sistemului și previne blocarea aplicației.
- **[PREPROCESS_INPUT] → [RN_INFERENCE]:** Tranziție condiționată de succesul încărcării resurselor (model + preprocesor).

---

### 4. Scheletul Complet al celor 3 Module Cerute

Toate cele 3 module pornesc și rulează fără erori.

#### **Modul 1: Data Logging / Acquisition**
* **Fișier:** `src/generare_date.py`
* **Funcționalitate:** Generează dataset-ul sintetic pe baza parametrilor fizici (viteză, trafic, penalizări).
* **Status:** [x] Funcțional, produce `data/raw/delivery_data.csv` cu 5000 observații.

#### **Modul 2: Neural Network Module**
* **Fișier:** `src/antrenare_retea.py`
* **Funcționalitate:** Definește arhitectura Keras (Dense 64 -> Dense 32 -> Dense 1), compilează modelul și îl antrenează. Salvează modelul în `models/model_livrare.keras`.
* **Status:** [x] Funcțional, modelul este definit, antrenat (MAE 6.45) și salvat.

#### **Modul 3: Web Service / UI**
* **Fișier:** `src/demo_live.py`
* **Funcționalitate:** Interfață consolă interactivă. Primește input de la utilizator, validează datele, apelează preprocesorul (`models/preprocessor.pkl`) și modelul pentru inferență.
* **Status:** [x] Funcțional, demonstrează pipeline-ul complet (Input -> Predicție).

---

## Structura Repository-ului la Finalul Etapei 4

```

project-name/
├── data/
│   ├── raw/                 \# CSV generat (delivery\_data.csv)
│   ├── train/               \# .npy files
│   ├── validation/          \# .npy files
│   └── test/                \# .npy files
├── models/
│   ├── model\_livrare.keras  \# Modelul Keras salvat
│   └── preprocessor.pkl     \# Scaler-ul salvat
├── src/
│   ├── generare\_date.py     \# Modul 1: Data Acquisition
│   ├── analiza\_eda.py       \# (Din Etapa 3)
│   ├── preprocesare\_finala.py \# (Din Etapa 3)
│   ├── antrenare\_retea.py   \# Modul 2: Neural Network
│   └── demo\_live.py         \# Modul 3: UI
├── docs/
│   ├── state\_machine.png    \# Diagrama SM (vezi instrucțiunile)
│   └── grafic\_antrenare.png \# Dovada antrenării
├── README.md
└── README\_Etapa4\_Arhitectura\_SIA\_03.12.2025.md

```


## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [x] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [x] Cod generare/achiziție date funcțional și documentat
- [x] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [x] Diagrama State Machine creată și salvată în `docs/`
- [x] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [x] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/generare_date.py`)
- [x] Produce minimum 40% date originale din dataset-ul final (100% în acest caz)
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3

### Modul 2: Neural Network
- [x] Arhitectură RN definită și documentată în cod (docstring detaliat)
- [x] README în `src/` sau comentarii detaliate în cod

### Modul 3: Web Service / UI
- [x] Propunere Interfață ce pornește fără erori (`python src/demo_live.py`)
- [x] Screenshot demonstrativ în `docs/`
```