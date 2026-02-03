import customtkinter as ctk
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys

# --- IMPORTURI NECESARE PENTRU EXE ---
import sklearn.compose
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.tree
import sklearn.utils

# --- CONFIGURĂRI VIZUALE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AplicatieLivrare(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Fereastra ---
        self.title("Sistem AI Predicție Livrare")
        self.geometry("450x700") # Am mărit puțin înălțimea pentru noul buton
        self.resizable(True, True)

        # Titlu
        self.label_titlu = ctk.CTkLabel(self, text="🚀 Estimator Livrare AI", font=("Arial", 22, "bold"))
        self.label_titlu.pack(pady=10)

        # --- ScrollableFrame ---
        self.frame = ctk.CTkScrollableFrame(self, label_text="Date Comandă")
        self.frame.pack(pady=10, padx=20, fill="both", expand=True)

        # 1. Distanța
        self.add_label("1. Distanța (km):")
        self.entry_distanta = ctk.CTkEntry(self.frame, placeholder_text="Ex: 12.5")
        self.entry_distanta.pack(pady=5, padx=10, fill="x")

        # 2. Trafic
        self.add_label("2. Nivel Trafic (1-5):")
        self.combo_trafic = ctk.CTkComboBox(self.frame, values=["1", "2", "3", "4", "5"])
        self.combo_trafic.set("3")
        self.combo_trafic.pack(pady=5, padx=10, fill="x")

        # 3. Ora
        self.add_label("3. Ora Comenzii (Ex: 14.5):")
        self.entry_ora = ctk.CTkEntry(self.frame, placeholder_text="14.5")
        self.entry_ora.pack(pady=5, padx=10, fill="x")

        # 4. Ziua
        self.add_label("4. Ziua Săptămânii:")
        self.combo_ziua = ctk.CTkComboBox(self.frame, values=['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică'])
        self.combo_ziua.set("Luni")
        self.combo_ziua.pack(pady=5, padx=10, fill="x")

        # 5. Vehicul
        self.add_label("5. Tip Vehicul:")
        self.combo_vehicul = ctk.CTkComboBox(self.frame, values=['Bicicletă', 'Scuter', 'Dubiță', 'Camion'])
        self.combo_vehicul.set("Scuter")
        self.combo_vehicul.pack(pady=5, padx=10, fill="x")

        # 6. Grad Încărcare
        self.add_label("6. Grad Încărcare (0.1 - 1.0):")
        self.slider_grad = ctk.CTkSlider(self.frame, from_=0.1, to=1.0, number_of_steps=9)
        self.slider_grad.set(0.5)
        self.slider_grad.pack(pady=5, padx=10, fill="x")

        # --- ZONA DE BUTOANE ---
        # Butonul Verde (Calculează)
        self.btn_predict = ctk.CTkButton(self, text="CALCULEAZĂ TIMPUL", command=self.fa_predictia, height=45, font=("Arial", 14, "bold"), fg_color="green")
        self.btn_predict.pack(pady=(10, 5), padx=20, fill="x")

        # Butonul Gri (Resetare) - NOU!
        self.btn_reset = ctk.CTkButton(self, text="RESETARE CÂMPURI", command=self.reseteaza_formular, height=35, font=("Arial", 12), fg_color="#555555", hover_color="#333333")
        self.btn_reset.pack(pady=(5, 20), padx=20, fill="x")

        # --- REZULTAT ---
        self.label_rezultat = ctk.CTkLabel(self, text="Introdu datele...", font=("Arial", 18), text_color="yellow")
        self.label_rezultat.pack(pady=(0, 10))

        # Încărcare modele la final
        self.incarca_modele()

    def add_label(self, text):
        label = ctk.CTkLabel(self.frame, text=text, anchor="w")
        label.pack(pady=(10, 0), padx=10, fill="x")

    def incarca_modele(self):
        try:
            path_model = 'models/optimized_model.keras'
            path_scaler = 'models/preprocessor.pkl'
            
            if not os.path.exists(path_model):
                base_path = os.path.dirname(sys.executable)
                path_model = os.path.join(base_path, 'models', 'optimized_model.keras')
                path_scaler = os.path.join(base_path, 'models', 'preprocessor.pkl')

            self.model = tf.keras.models.load_model(path_model)
            self.preprocessor = joblib.load(path_scaler)
            print("Modele încărcate cu succes!")
        except Exception as e:
            self.label_rezultat.configure(text=f"EROARE MODEL:\n{str(e)}", text_color="red")

    def fa_predictia(self):
        try:
            dist = float(self.entry_distanta.get())
            trafic = int(self.combo_trafic.get())
            ora = float(self.entry_ora.get())
            ziua = self.combo_ziua.get()
            vehicul = self.combo_vehicul.get()
            grad = self.slider_grad.get()

            date = {
                'Distanța (km)': [dist],
                'Nivel trafic': [trafic],
                'Ora livrării': [ora],
                'Ziua săptămânii': [ziua],
                'Tip vehicul': [vehicul],
                'Grad de încărcare': [grad]
            }
            df = pd.DataFrame(date)
            
            X_input = self.preprocessor.transform(df)
            minute = self.model.predict(X_input, verbose=0)[0][0]
            
            self.label_rezultat.configure(text=f"⏱️ Timp Estimat: {minute:.0f} minute", text_color="#00FF00")
            
        except ValueError:
            self.label_rezultat.configure(text="❌ Verifică numerele introduse!", text_color="red")
        except Exception as e:
            self.label_rezultat.configure(text=f"Eroare: {str(e)}", text_color="red")

    # --- FUNCȚIA NOUĂ DE RESETARE ---
    def reseteaza_formular(self):
        # 1. Ștergem textul
        self.entry_distanta.delete(0, 'end')
        self.entry_ora.delete(0, 'end')
        
        # 2. Resetăm meniurile la valorile implicite
        self.combo_trafic.set("3")
        self.combo_ziua.set("Luni")
        self.combo_vehicul.set("Scuter")
        
        # 3. Resetăm sliderul la mijloc
        self.slider_grad.set(0.5)
        
        # 4. Resetăm mesajul de rezultat
        self.label_rezultat.configure(text="Câmpuri resetate!", text_color="white")

if __name__ == "__main__":
    app = AplicatieLivrare()
    app.mainloop()