# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Radu Mihaita-Gheorghe 
**Link Repository GitHub:** https://github.com/Mihai0107/Proiect-RN.git
**Data predării:** 11.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/raw/` (100% generate)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

*Notă: Deoarece întregul dataset a fost generat unitar în Etapa 4 (5000 samples), nu este necesară o combinare suplimentară. Datele au fost preprocesate consistent folosind `src/preprocesare_finala.py`.*

**Verificare rapidă:**
Datele sunt deja împărțite și salvate în formatele `.npy` în folderele `data/train`, `data/validation`, `data/test`.

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (100% originale, 5000 observații).
2. **Minimum 10 epoci**, batch size 32.
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%.
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU).
5. **Metrici calculate pe test set:**
   *Notă: Proiectul fiind de tip Regresie (predicție timp continuu), raportăm MAE și R2.*
   - **MAE (Mean Absolute Error):** 6.4563 minute
   - **MSE (Mean Squared Error):** 290.6815
   - **R2 Score:** 0.9741 (Excelent - explică 97.4% din variație)
6. **Salvare model antrenat** în `models/trained_model.keras`.
7. **Integrare în UI din Etapa 4:**
   - UI încarcă modelul ANTRENAT.
   - Inferență REALĂ demonstrată.
   - Screenshot în `docs/screenshots/inference_real.png`.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru Adam optimizer; asigură o convergență stabilă fără a oscila excesiv în jurul minimului global. |
| Batch size | 32 | Compromis optim între viteza de execuție și stabilitatea gradientului pentru un dataset de 5000 linii. |
| Number of epochs | 100 | S-a utilizat Early Stopping (patience=10), modelul oprindu-se automat când `val_loss` nu a mai scăzut, prevenind overfitting-ul. |
| Optimizer | Adam | Algoritm adaptiv eficient pentru date tabulare și regresie, converge mai rapid decât SGD clasic. |
| Loss function | MSE (Mean Squared Error) | Funcție standard pentru regresie care penalizează pătratic erorile mari (outlierii), forțând modelul să fie precis pe cazurile extreme. |
| Activation functions | ReLU (Hidden), Linear (Output) | ReLU rezolvă problema vanishing gradient; Output Linear este **obligatoriu** pentru regresie (ne permite să prezicem orice valoare pozitivă, ex: 45.5 min). |

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - activat (patience=10 pe `val_loss`).
2. **Grafic loss și val_loss** salvat în `docs/loss_curve.png`. Arată convergența clară și lipsa overfitting-ului major.
3. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2).

---

### Nivel 3 – Bonus (până la 100%)

**Activitate realizată:** Compararea a 2 arhitecturi diferite.

Am antrenat un al doilea model (**Model Bonus**) cu o arhitectură mult mai complexă (mai mulți neuroni, mai multe straturi și Dropout) pentru a verifica dacă creșterea capacității de învățare reduce eroarea.

**Tabel Comparativ:**

| Caracteristică | **Model Standard (Nivel 1)** | **Model Bonus (Nivel 3)** |
| :--- | :--- | :--- |
| **Arhitectură** | Dense(64) -> Dense(32) -> Out(1) | Dense(128) -> Dropout(0.2) -> Dense(64) -> Dense(32) -> Out(1) |
| **Complexitate** | Redusă (Rapid și Eficient) | Ridicată (Mai lent, risc de overfitting) |
| **MAE (Eroare Medie)** | **6.4563 min** | **6.6183 min** |
| **Concluzie** | **Mai performant** | Mai slab (Overfitting) |

**Justificare Alegere Finală:**
Rezultatele experimentale arată că **Modelul Bonus a obținut o eroare mai mare** (+0.16 minute) față de cel standard. Acest lucru indică faptul că arhitectura complexă a suferit de **Overfitting** (a început să memoreze zgomotul din datele de antrenare), nefiind capabilă să generalizeze la fel de bine pe datele de test.

Prin urmare, am decis să păstrăm **Modelul Standard** pentru producție, deoarece este:
1. Mai precis (MAE mai mic).
2. Mai rapid la inferență.
3. Mai puțin predispus la erori pe date noi.

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența respectă fluxul din State Machine-ul definit în Etapa 4.

**Implementare concretă pentru Predicție Livrare:**

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `LOAD_RESOURCES` | Încărcare model `models/trained_model.keras` și scaler `models/preprocessor.pkl` |
| `PREPROCESS_INPUT` | Transformare date user (StandardScaler + OneHot) folosind pipeline-ul salvat |
| `RN_INFERENCE` | Forward pass `model.predict()` cu modelul ANTRENAT |
| `DISPLAY_RESULT` | Afișare timp estimat (ex: "45 minute") în consolă |

**În `src/demo_live.py` (UI actualizat):**
Modelul dummy a fost înlocuit cu:
```python
model = tf.keras.models.load_model('models/trained_model.keras')
prediction = model.predict(X_input)  # predicție REALĂ bazată pe distanță și trafic