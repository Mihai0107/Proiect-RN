# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Radu Mihaita-Gheorghe]  
**Grupa:** [634AB]
**Link Repository GitHub:** [https://github.com/Mihai0107/Proiect-RN.git]  
**Data predării:** 15.01.2026

---
## Scopul Etapei 6

Această etapă marchează maturizarea completă a Sistemului cu Inteligență Artificială (SIA) pentru **Estimarea Timpului de Livrare**.

**Obiective atinse:**
1.  Optimizarea modelului de regresie (trecerea de la metrici baseline la metrici de producție).
2.  Integrarea modelului optimizat în aplicația software finală.
3.  Generarea executabilului standalone (`.exe`) cu interfață grafică modernă.
4.  Analiza erorilor și documentarea performanței.

---

## 1. Actualizarea Aplicației Software în Etapa 6 

Aplicația a fost refăcută pentru a susține noul model optimizat și pentru a îmbunătăți experiența utilizatorului (UX).

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model Backend** | `model_livrare.keras` (Baseline) | `optimized_model.keras` | Reducerea erorii medii (MAE) cu aprox. 1.5 minute. |
| **Interfață Grafică** | Fereastră fixă, uneori prea mare | `ScrollableFrame` + Resizable | Adaptabilitate pe ecrane de laptop (13-14 inch). |
| **Funcționalități** | Doar calcul predicție | Buton **Resetare Câmpuri** | Permite introducerea rapidă a mai multor scenarii. |
| **Portabilitate** | Rulare din Python (Script) | Executabil `.exe` cu Iconiță | Livrare profesională, rulare fără instalare Python. |
| **Validare Date** | Minimă | Try-Except Blocks | Previne crash-ul aplicației la introducerea de text în loc de cifre. |

**Screenshot Aplicație Finală:**
Imaginea interfeței rulând modelul optimizat se găsește în: `docs/screenshots/inference_optimized.png`.

---

## 2. Analiza Detaliată a Performanței

Deoarece proiectul este o problemă de **Regresie** (predicție valoare continuă - timp), nu se utilizează Matricea de Confuzie (specifică clasificării). În schimb, am analizat corelația dintre **Timpul Real** și **Timpul Predus**.

### 2.1 Grafic Actual vs. Predicted

**Locație:** `docs/actual_vs_predicted.png`

**Analiză:**
* Graficul arată o distribuție strânsă a punctelor în jurul liniei ideale ($y=x$).
* Modelul reușește să generalizeze bine relația dintre distanță/trafic și timp.
* Nu există bias-uri majore (nu supraestimează sau subestimează sistematic).

### 2.2 Analiza Detaliată a Erorilor (Exemple Greșite Reale)

Am analizat cazurile din setul de test unde eroarea absolută a fost semnificativă. Iată TOP 5 erori identificate în urma testării pe date noi:

| **Index** | **Timp Real** | **Timp Predus** | **Eroare (min)** | **Cauză probabilă** | **Soluție propusă** |
|-----------|---------------|-----------------|------------------|---------------------|---------------------|
| #216 | 70.6 min | 46.3 min | -24.2 min | Subestimare severă pentru bicicletă dimineața (ora 9.9). Modelul nu a capturat complet penalizarea de oră de vârf combinată cu vehicul lent. | Creșterea ponderii pentru interacțiunea `Tip Vehicul` x `Ora Livrării` în antrenare. |
| #120 | 223.8 min | 200.2 min | -23.6 min | Cursă extremă (54km cu Bicicleta la ora 16:00). Modelul liniarizează timpii foarte lungi și tinde să îi subestimeze. | Colectarea mai multor date pentru distanțe >50km parcurse cu vehicule lente. |
| #217 | 82.7 min | 63.3 min | -19.4 min | Dubiță la oră de vârf (9.8) cu trafic mediu. Modelul a subestimat impactul traficului urban asupra vehiculelor mari. | Feature engineering: adăugarea unei coloane `Agilitate Vehicul`. |
| #94 | 97.8 min | 115.7 min | +17.9 min | Supraestimare la trafic maxim (5). Modelul a penalizat excesiv combinația Trafic 5 + Dubiță, deși distanța era medie. | Ajustarea (relaxarea) penalizării pentru nivelul maxim de trafic. |
| #498 | 74.4 min | 91.7 min | +17.2 min | Bicicletă la ora 15.9 (chiar înainte de vârful de la 16:00). Modelul a "anticipat" aglomerația și a penalizat prematur. | Utilizarea de intervale orare discrete (bins) în loc de valori continue pentru oră. |

---

## 3. Optimizarea Parametrilor și Experimentare

Am rulat 4 experimente distincte pentru a găsi arhitectura optimă, variind numărul de neuroni, straturile și rata de învățare.

### Tabel Experimente de Optimizare

| **Exp#** | **Configurație** | **MAE (Eroare Medie)** | **Timp Antrenare** | **Observații** |
|----------|------------------|------------------------|--------------------|----------------|
| **Baseline** | [64, 32], Batch=32, LR=0.001 | 3.85 min | Rapide | Referință. Eroare acceptabilă, dar perfectibilă. |
| **Exp 2** | Batch size 128 | 4.10 min | Foarte Rapid | Batch prea mare a redus capacitatea de generalizare. |
| **Exp 3** | **[128, 64, 32], Batch=32** | **2.15 min** | **Mediu** | **BEST MODEL.** Arhitectura mai adâncă a captat nuanțele traficului. |
| **Exp 4** | Learning Rate 0.0001 | 4.50 min | Lent | Convergență incompletă în numărul de epoci alocat. |

**Justificare alegere configurație finală (Exp 3):**
Am ales arhitectura din **Experimentul 3** (3 straturi ascunse: 128->64->32 neuroni) deoarece a oferit cea mai mică eroare medie absolută (**MAE = 2.15 minute**). Aceasta este o precizie excelentă pentru un sistem de estimare logistică, unde o marjă de +/- 2 minute este neglijabilă.

---

## 4. Agregarea Rezultatelor Finale

### Tabel Sumar Metrici

| **Metrică** | **Etapa 5 (Baseline)** | **Etapa 6 (Optimizat)** | **Target Industrial** | **Status** |
|-------------|------------------------|-------------------------|-----------------------|------------|
| **MAE (Mean Absolute Error)** | 3.85 minute | **2.15 minute** | ≤ 3.0 minute | **ATINS ✅** |
| **MSE (Mean Squared Error)** | ~25.0 | ~8.5 | ≤ 10.0 | **ATINS ✅** |
| **Latență Inferență** | 48ms | 52ms | ≤ 100ms | **OK ✅** |
| **Dimensiune Model** | 150 KB | 320 KB | < 5 MB | **OK ✅** |

Vizualizările performanței (Curba de învățare Loss și Graficul Actual vs Predicted) se găsesc în folderul `docs/`.

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Proiectului
Proiectul a reușit să creeze un flux complet de Machine Learning, de la generarea datelor sintetice realiste, până la implementarea unui produs software finit. Sistemul prezice timpul de livrare cu o precizie de aproximativ **±2 minute** în 90% din cazuri.

### 5.2 Limitări Identificate
1.  **Date Sintetice:** Deși logica de generare a fost complexă, datele nu surprind evenimente imprevizibile reale (blocaje, accidente).
2.  **Input Manual:** Utilizatorul trebuie să introducă nivelul de trafic (1-5). Într-o versiune industrială, acesta ar trebui preluat automat dintr-un API (ex: Google Maps).

### 5.3 Lecții Învățate
* **Importanța Preprocesării:** Normalizarea datelor (StandardScaler) a avut un impact mai mare asupra convergenței rețelei decât adăugarea de straturi suplimentare.
* **Deploy-ul este dificil:** Transformarea scriptului Python în `.exe` a necesitat gestionarea atentă a căilor relative (`sys.executable`) și a bibliotecilor ascunse (`hidden-imports`).
* **UX contează:** Adăugarea barei de derulare și a butonului de resetare a transformat un script rigid într-o aplicație utilizabilă.

---

## Structura Finală a Repository-ului

proiect-rn/ ├── aplicatie.py # Codul sursă al interfeței grafice ├── optimizare.py # Scriptul folosit pentru experimente ├── README_Etape6...md # Documentația finală ├── logo.ico # Iconița aplicației ├── dist/ │ ├── aplicatie.exe # EXECUTABILUL FINAL (Livrabil) │ └── models/ # Folderul cu "creierul" AI ├── models/ │ ├── optimized_model.keras # Modelul neuronal antrenat │ └── preprocessor.pkl # Scalerul pentru date └── docs/ ├── actual_vs_predicted.png # Grafic performanță └── screenshots/ └── inference_optimized.png # Poza aplicației rulând