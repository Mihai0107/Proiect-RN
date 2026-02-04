import customtkinter as ctk
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys


# Fara importurile astea explicite, executabilul nu gaseste sklearn si da eroare la rulare
import sklearn.compose
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.tree
import sklearn.utils

# --- CONFIGURĂRI DESIGN ---
ctk.set_appearance_mode("Dark") # Arata mai bine pe Dark Mode
ctk.set_default_color_theme("blue")

class AplicatieLivrare(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configurare Fereastra Principala ---
        self.title("Sistem AI Predicție Livrare")
        # Am pus 700 inaltime ca sa fie loc si pentru butonul de Reset
        self.geometry("450x700") 
        self.resizable(True, True)

        # Titlu mare sus
        self.label_titlu = ctk.CTkLabel(self, text="🚀 Estimator Livrare AI", font=("Arial", 22, "bold"))
        self.label_titlu.pack(pady=10)

        # --- Zona Scrollabila ---
        # Folosesc ScrollableFrame pentru ca pe laptopuri mici campurile ieseau din ecran
        self.frame = ctk.CTkScrollableFrame(self, label_text="Date Comandă")
        self.frame.pack(pady=10, padx=20, fill="both", expand=True)

        # --- INPUTURI ---
        
        # 1. Distanța
        self.add_label("1. Distanța (km):")
        self.entry_distanta = ctk.CTkEntry(self.frame, placeholder_text="Ex: 12.5")
        self.entry_distanta.pack(pady=5, padx=10, fill="x")

        # 2. Trafic (Dropdown)
        self.add_label("2. Nivel Trafic (1-5):")
        self.combo_trafic = ctk.CTkComboBox(self.frame, values=["1", "2", "3", "4", "5"])
        self.combo_trafic.set("3") # Default punem mediu
        self.combo_trafic.pack(pady=5, padx=10, fill="x")

        # 3. Ora
        self.add_label("3. Ora Comenzii (Ex: 14.5):")
        self.entry_ora = ctk.CTkEntry(self.frame, placeholder_text="14.5")
        self.entry_ora.pack(pady=5, padx=10, fill="x")

        # 4. Ziua
        self.add_label("4. Ziua Săptămânii:")
        zile = ['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică']
        self.combo_ziua = ctk.CTkComboBox(self.frame, values=zile)
        self.combo_ziua.set("Luni")
        self.combo_ziua.pack(pady=5, padx=10, fill="x")

        # 5. Vehicul
        self.add_label("5. Tip Vehicul:")
        self.combo_vehicul = ctk.CTkComboBox(self.frame, values=['Bicicletă', 'Scuter', 'Dubiță', 'Camion'])
        self.combo_vehicul.set("Scuter")
        self.combo_vehicul.pack(pady=5, padx=10, fill="x")

        # 6. Slider pentru incarcare
        self.add_label("6. Grad Încărcare (0.1 - 1.0):")
        self.slider_grad = ctk.CTkSlider(self.frame, from_=0.1, to=1.0, number_of_steps=9)
        self.slider_grad.set(0.5)
        self.slider_grad.pack(pady=5, padx=10, fill="x")

        # --- BUTOANE DE CONTROL ---
        
        # Butonul Verde - Actioneaza predictia
        self.btn_predict = ctk.CTkButton(self, text="CALCULEAZĂ TIMPUL", command=self.fa_predictia, height=45, font=("Arial", 14, "bold"), fg_color="green")
        self.btn_predict.pack(pady=(10, 5), padx=20, fill="x")

        # Butonul Gri - Curata tot formularul (UX feature)
        self.btn_reset = ctk.CTkButton(self, text="RESETARE CÂMPURI", command=self.reseteaza_formular, height=35, font=("Arial", 12), fg_color="#555555", hover_color="#333333")
        self.btn_reset.pack(pady=(5, 20), padx=20, fill="x")

        # --- ZONA DE AFISARE REZULTAT ---
        self.label_rezultat = ctk.CTkLabel(self, text="Introdu datele...", font=("Arial", 18), text_color="yellow")
        self.label_rezultat.pack(pady=(0, 10))

        # La final, incarcam logica AI
        self.incarca_modele()

    def add_label(self, text):
        # Helper ca sa nu scriu cod duplicat pentru fiecare label
        label = ctk.CTkLabel(self.frame, text=text, anchor="w")
        label.pack(pady=(10, 0), padx=10, fill="x")

    def incarca_modele(self):
        """
        Incarca modelul .keras si scaler-ul .pkl
        Gestioneaza problema cailor relative pentru cand rulam din .exe
        """
        try:
            # Cai implicite (pentru rulare din IDE/Terminal)
            path_model = 'models/optimized_model.keras'
            path_scaler = 'models/preprocessor.pkl'
            
            # Verificam daca rulam ca executabil (PyInstaller face folder temporar _MEI...)
            if not os.path.exists(path_model):
                # print("Rulare din EXE detectata, ajustam caile...") 
                base_path = os.path.dirname(sys.executable)
                path_model = os.path.join(base_path, 'models', 'optimized_model.keras')
                path_scaler = os.path.join(base_path, 'models', 'preprocessor.pkl')

            # print(f"Incarcare model din: {path_model}")
            self.model = tf.keras.models.load_model(path_model)
            self.preprocessor = joblib.load(path_scaler)
            print("--- Sistem AI Initializat cu succes ---")
            
        except Exception as e:
            # Daca crapa aici, afisam eroarea in interfata sa stim ce lipseste
            self.label_rezultat.configure(text=f"EROARE FATALA:\n{str(e)}", text_color="red")
            print(f"Eroare incarcare: {e}")

    def fa_predictia(self):
        try:
            # 1. Preluam datele brute din interfata
            # print("Preluare date user...") 
            dist = float(self.entry_distanta.get())
            trafic = int(self.combo_trafic.get())
            ora = float(self.entry_ora.get())
            ziua = self.combo_ziua.get()
            vehicul = self.combo_vehicul.get()
            grad = self.slider_grad.get()

            # 2. Construim DataFrame-ul exact cum a fost la antrenare (aceleasi coloane)
            date = {
                'Distanța (km)': [dist],
                'Nivel trafic': [trafic],
                'Ora livrării': [ora],
                'Ziua săptămânii': [ziua],
                'Tip vehicul': [vehicul],
                'Grad de încărcare': [grad]
            }
            df = pd.DataFrame(date)
            
            # 3. Preprocesare (Scalare + OneHotEncoding)
            X_input = self.preprocessor.transform(df)
            
            # 4. Inferenta (Predictia efectiva)
            # verbose=0 ca sa nu umplem consola de logs inutile
            minute = self.model.predict(X_input, verbose=0)[0][0]
            
            # 5. Afisare rezultat user-friendly
            self.label_rezultat.configure(text=f"⏱️ Timp Estimat: {minute:.0f} minute", text_color="#00FF00")
            print(f"Predictie realizata: {minute:.2f} min pentru distanta {dist}km")
            
        except ValueError:
            # Daca userul baga litere la distanta sau ora
            self.label_rezultat.configure(text="❌ Verifică dacă ai introdus numere corecte!", text_color="red")
        except Exception as e:
            self.label_rezultat.configure(text=f"Eroare interna: {str(e)}", text_color="red")

    def reseteaza_formular(self):
        """Sterge campurile pentru a introduce o comanda noua rapid"""
        # print("Resetare formular...")
        
        # Stergem textul
        self.entry_distanta.delete(0, 'end')
        self.entry_ora.delete(0, 'end')
        
        # Resetam meniurile la valorile implicite (cele mai comune)
        self.combo_trafic.set("3")
        self.combo_ziua.set("Luni")
        self.combo_vehicul.set("Scuter")
        
        # Slider la mijloc
        self.slider_grad.set(0.5)
        
        self.label_rezultat.configure(text="Câmpuri resetate!", text_color="white")

if __name__ == "__main__":
    app = AplicatieLivrare()
    app.mainloop()