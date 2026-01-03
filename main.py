import sys
import os
import cv2
import time
import hashlib
import csv
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                             QMessageBox, QFrame, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# =============================================================================
# 1. LOGIQUE MÉTIER : ROULETTE & ENTROPIE
# =============================================================================

class RouletteLogic:
    # Définition des numéros rouges (Standard Européen)
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}

    @staticmethod
    def get_color_and_label(number: int):
        """Retourne la couleur et le texte formaté pour l'UI."""
        if number == 0:
            return "#008000", "VERT" # Vert Zéro
        elif number in RouletteLogic.RED_NUMBERS:
            return "#cc0000", "ROUGE"
        else:
            return "#000000", "NOIR"

class EntropyEngine:
    @staticmethod
    def sha512(data: bytes) -> bytes:
        return hashlib.sha512(data).digest()

    @staticmethod
    def get_roulette_spin(frame):
        """
        1. Capture l'entropie
        2. Génère un float 0.0-1.0
        3. Transforme en entier 0-36 (Projection)
        """
        if frame is None: return None
        
        # --- Etape 1 : Entropie Visuelle ---
        h, w = frame.shape[:2]
        # Crop central pour éviter les bords statiques
        nh, nw = int(h * 0.75), int(w * 0.75)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        cropped = frame[y0:y0+nh, x0:x0+nw]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # --- Etape 2 : Entropie Temporelle ---
        t_ns = time.time_ns()
        
        # --- Etape 3 : Mixage Cryptographique ---
        raw_data = gray.tobytes() + t_ns.to_bytes(8, "big", signed=False)
        hashed = EntropyEngine.sha512(raw_data)
        
        # --- Etape 4 : Conversion en nombre Roulette (0-36) ---
        int_val = int.from_bytes(hashed[:8], "big")
        float_val = int_val / (2**64 - 1)
        
        # Projection sur 37 segments (0 à 36)
        roulette_number = int(float_val * 37)
        
        color_hex, color_name = RouletteLogic.get_color_and_label(roulette_number)

        return {
            "number": roulette_number,
            "color_hex": color_hex,
            "color_name": color_name,
            "raw_float": float_val
        }

# =============================================================================
# 2. WORKER VIDEO (THREAD SAFE)
# =============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True
        self.cap = None
        self.current_frame = None # Variable tampon thread-safe

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.current_frame = frame.copy() # Copie de sécurité
            self.change_pixmap_signal.emit(frame)
            time.sleep(0.030) # ~30 FPS

        self.cap.release()

    def get_safe_frame(self):
        if self.current_frame is not None:
            return self.current_frame
        return None

    def stop(self):
        self._run_flag = False
        self.wait()

# =============================================================================
# 3. INTERFACE CASINO PRO (PyQt6)
# =============================================================================

class CasinoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KALI-ROULETTE : Générateur Entropique")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #121212; color: white;") 

        # Vérification Vidéo
        self.video_file = "kalicasino.mp4"
        if not os.path.exists(self.video_file):
            QMessageBox.critical(self, "Erreur", f"Vidéo manquante : {self.video_file}")
            sys.exit(1)

        # Layout Principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # =================================================
        # --- COLONNE GAUCHE : Moteur Vidéo & Contrôles ---
        # =================================================
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Titre
        self.lbl_title_video = QLabel("SOURCE D'ENTROPIE")
        self.lbl_title_video.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_title_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.lbl_title_video)

        # Image Vidéo
        self.image_label = QLabel()
        self.image_label.setMinimumSize(480, 270)
        self.image_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.image_label)
        
        # --- SÉLECTEUR DE NOMBRE (SPINBOX) ---
        self.lbl_spin = QLabel("Nombre de tirages pour le test :")
        self.lbl_spin.setStyleSheet("margin-top: 10px; color: #ccc;")
        self.left_layout.addWidget(self.lbl_spin)

        self.spin_count = QSpinBox()
        self.spin_count.setRange(1000, 1000000) # De 1000 à 1 million
        self.spin_count.setSingleStep(1000)      # Pas de 1000
        self.spin_count.setValue(1000)           # Valeur par défaut
        self.spin_count.setStyleSheet("""
            QSpinBox { 
                background-color: #333; color: white; padding: 5px; font-size: 14px; border: 1px solid #555;
            }
        """)
        self.left_layout.addWidget(self.spin_count)
        # ---------------------------------------

        # Bouton Analyse
        self.btn_analyze = QPushButton("📊 LANCER L'ANALYSE")
        self.btn_analyze.setFixedHeight(50)
        self.btn_analyze.setStyleSheet("""
            QPushButton { background-color: #0066cc; color: white; font-weight: bold; border-radius: 5px; font-size: 14px; }
            QPushButton:hover { background-color: #0055aa; }
        """)
        self.btn_analyze.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.left_layout.addWidget(self.btn_analyze)

        # Barre de progression
        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar { border: 1px solid #444; text-align: center; color: white; } QProgressBar::chunk { background-color: #0066cc; }")
        self.left_layout.addWidget(self.progress)

        # =================================================
        # --- COLONNE DROITE : Tapis de Jeux (Roulette) ---
        # =================================================
        self.right_panel = QFrame()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        self.lbl_casino = QLabel("RÉSULTAT DU TIRAGE")
        self.lbl_casino.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.lbl_casino.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.lbl_casino)

        # Affichage Gros Numéro
        self.result_box = QLabel("?")
        self.result_box.setFont(QFont("Impact", 120))
        self.result_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_box.setStyleSheet("background-color: #222; color: #555; border-radius: 15px; border: 4px solid #444;")
        self.result_box.setFixedHeight(250)
        self.right_layout.addWidget(self.result_box)

        # Label Couleur
        self.lbl_color = QLabel("-")
        self.lbl_color.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.lbl_color.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.lbl_color)

        # Bouton SPIN
        self.btn_spin = QPushButton("LANCER LA BILLE")
        self.btn_spin.setFixedHeight(80)
        self.btn_spin.setStyleSheet("""
            QPushButton { background-color: #eebb00; color: black; font-size: 20px; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background-color: #ffcc00; }
            QPushButton:pressed { background-color: #ccaa00; }
        """)
        self.btn_spin.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_spin.clicked.connect(self.spin_wheel)
        self.right_layout.addWidget(self.btn_spin)

        # Ajout des panneaux au layout principal
        self.main_layout.addWidget(self.left_panel, 3) # 60% largeur
        self.main_layout.addWidget(self.right_panel, 2) # 40% largeur

        # --- Démarrage Thread Vidéo ---
        self.thread = VideoThread(self.video_file)
        self.thread.change_pixmap_signal.connect(self.update_video_display)
        self.thread.start()

    def update_video_display(self, cv_img):
        """Met à jour l'image vidéo dans l'interface."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def spin_wheel(self):
        """Action du bouton Spin : Récupère une frame et calcule le résultat."""
        frame = self.thread.get_safe_frame()
        if frame is None:
            self.lbl_color.setText("Erreur Vidéo")
            return

        result = EntropyEngine.get_roulette_spin(frame)
        
        # Mise à jour UI
        num = result['number']
        col_hex = result['color_hex']
        col_name = result['color_name']

        self.result_box.setText(str(num))
        self.result_box.setStyleSheet(f"background-color: {col_hex}; color: white; border-radius: 15px; border: 4px solid white;")
        self.lbl_color.setText(col_name)

    # =================================================
    # --- LOGIQUE D'ANALYSE OPTIMISÉE (BATCH) ---
    # =================================================

    def start_analysis(self):
        """Lance l'analyse avec le nombre choisi par l'utilisateur."""
        self.btn_analyze.setEnabled(False)
        self.btn_spin.setEnabled(False)
        self.progress.setValue(0)
        self.result_box.setText("...")
        self.result_box.setStyleSheet("background-color: #222; color: #fff; border: 4px solid #444; border-radius: 15px;")
        
        # On récupère la valeur du SpinBox
        self.TARGET_COUNT = self.spin_count.value()
        
        self.batch_data = []
        self.batch_count = 0
        
        # Timer rapide (10ms) pour appeler la fonction de calcul par paquets
        self.timer = QTimer()
        self.timer.timeout.connect(self._analysis_step_batch)
        self.timer.start(10)

    def _analysis_step_batch(self):
        """Génère par paquets pour aller vite même si on demande 100 000."""
        # On calcule par paquet de 50 pour garder l'interface fluide
        BATCH_SIZE = 50 
        
        for _ in range(BATCH_SIZE):
            if self.batch_count >= self.TARGET_COUNT:
                self.timer.stop()
                self._finalize_analysis()
                return

            frame = self.thread.get_safe_frame()
            if frame is not None:
                res = EntropyEngine.get_roulette_spin(frame)
                self.batch_data.append(res['number'])
                self.batch_count += 1
        
        # Mise à jour barre de progression
        progress_val = int((self.batch_count / self.TARGET_COUNT) * 100)
        self.progress.setValue(progress_val)

    def _finalize_analysis(self):
        """Fin de l'analyse : Sauvegarde CSV et Affichage Graphique."""
        self.btn_analyze.setEnabled(True)
        self.btn_spin.setEnabled(True)
        self.result_box.setText("FIN")
        self.progress.setValue(100)

        # 1. Sauvegarde CSV (Utile pour le prof)
        try:
            with open("analyse_roulette.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Tour", "Numero"])
                for idx, n in enumerate(self.batch_data):
                    writer.writerow([idx+1, n])
            print(f"[Info] CSV généré avec {len(self.batch_data)} lignes.")
        except Exception as e:
            print(f"[Erreur] Impossible d'écrire le CSV: {e}")
        
        # 2. Affichage Graphique
        self.show_graph()

    def show_graph(self):
        """Affiche l'analyse statistique complète (Histogramme + Courbe de stabilité)."""
        
        # Calcul des fréquences
        counts = [self.batch_data.count(i) for i in range(37)]
        # Moyenne théorique adaptée au nombre choisi (ex: 1000/37 ou 100000/37)
        expected = self.TARGET_COUNT / 37

        # Création de la fenêtre Matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.4)

        # --- GRAPHIQUE 1 : L'HISTOGRAMME ---
        bar_colors = []
        for i in range(37):
            if i == 0: bar_colors.append('green')
            elif i in RouletteLogic.RED_NUMBERS: bar_colors.append('red')
            else: bar_colors.append('black')

        ax1.bar(range(37), counts, color=bar_colors, edgecolor='gray', alpha=0.8)
        ax1.axhline(y=expected, color='blue', linestyle='--', linewidth=2, label=f'Moyenne Théorique ({expected:.1f})')
        ax1.set_title(f"Répartition brute sur {self.TARGET_COUNT} tirages")
        ax1.set_ylabel("Nombre de sorties")
        ax1.set_xlabel("Numéros Roulette (0-36)")
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # --- GRAPHIQUE 2 : LA COURBE DE STABILITÉ (Triée) ---
        sorted_counts = sorted(counts)
        
        ax2.plot(range(37), sorted_counts, color='purple', linewidth=3, marker='o', markersize=4, label='Répartition Triée')
        ax2.axhline(y=expected, color='blue', linestyle='--', linewidth=2, label='Objectif Idéal (Plat)')
        
        # Zone de tolérance (+/- 20%)
        ax2.fill_between(range(37), expected * 0.8, expected * 1.2, color='blue', alpha=0.1, label='Zone de Tolérance Acceptable')

        ax2.set_title("Courbe de Régularité (Doit tendre vers le plat avec un grand nombre)")
        ax2.set_ylabel("Fréquence")
        ax2.set_xlabel("Index trié (du moins sorti au plus sorti)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        print("[Info] Graphique Matplotlib généré.")
        plt.show()
    
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CasinoApp()
    window.show()
    sys.exit(app.exec())