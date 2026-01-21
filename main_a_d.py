import sys
import os
import cv2
import time
import hashlib
import random
import csv  # Pour sauvegarder l'historique
import numpy as np
import matplotlib.pyplot as plt
import threading
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QMessageBox, QFrame, QSpinBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# =============================================================================
# 1. MOTEUR CRYPTO (SHA-512) + TIRAGE UNIFORME (Rejection Sampling)
# =============================================================================

class RouletteLogic:
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}

    @staticmethod
    def get_info(number: int):
        if number == 0:
            return "#00aa00", "ZÉRO (VERT)"
        elif number in RouletteLogic.RED_NUMBERS:
            return "#e60000", "ROUGE"
        else:
            return "#111111", "NOIR"


class EntropyEngine:
    _counter = 0
    _last_selector = None
    _prev_hash = b"\x00" * 64
    _U64_MAX_PLUS_1 = 1 << 64
    _RANGE = 37
    _LIMIT = (_U64_MAX_PLUS_1 // _RANGE) * _RANGE

    @staticmethod
    def _uniform_0_36_from_digest(digest: bytes) -> int:
        for i in range(0, 64, 8):
            x = int.from_bytes(digest[i:i+8], "big", signed=False)
            if x < EntropyEngine._LIMIT:
                return x % EntropyEngine._RANGE
        digest2 = hashlib.sha512(digest).digest()
        for i in range(0, 64, 8):
            x = int.from_bytes(digest2[i:i+8], "big", signed=False)
            if x < EntropyEngine._LIMIT:
                return x % EntropyEngine._RANGE
        return int.from_bytes(digest2[:8], "big", signed=False) % EntropyEngine._RANGE

    @staticmethod
    def get_spin(frame):
        if frame is None:
            return None

        EntropyEngine._counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        t_ns = time.time_ns()

        selector = (t_ns ^ (EntropyEngine._counter * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        if selector == EntropyEngine._last_selector:
            selector ^= 0xD1B54A32D192ED03
        EntropyEngine._last_selector = selector

        patch_size = 128
        max_x = max(1, w - patch_size)
        max_y = max(1, h - patch_size)
        x0 = selector % max_x
        y0 = (selector >> 16) % max_y
        patch = gray[y0:y0 + patch_size, x0:x0 + patch_size]

        n_points = 4096
        total = h * w
        rng = np.random.default_rng(selector)
        idx = rng.choice(total, size=min(n_points, total), replace=False)
        ys = (idx // w).astype(np.int32)
        xs = (idx % w).astype(np.int32)
        pixels = gray[ys, xs]

        raw_data = (
            patch.tobytes() + pixels.tobytes()
            + t_ns.to_bytes(16, "big", signed=False)
            + EntropyEngine._counter.to_bytes(8, "big", signed=False)
            + EntropyEngine._prev_hash
        )

        hashed = hashlib.sha512(raw_data).digest()
        EntropyEngine._prev_hash = hashed
        number = EntropyEngine._uniform_0_36_from_digest(hashed)

        col_hex, col_name = RouletteLogic.get_info(number)
        return {"number": number, "hex": col_hex, "name": col_name}


# =============================================================================
# 2. THREAD VIDÉO
# =============================================================================

class VideoThread(QThread):
    pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.running = True
        self.current_frame = None
        self._lock = threading.Lock()
        self._frame_buffer = deque(maxlen=120)

    def run(self):
        cap = cv2.VideoCapture(self.path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.current_frame = frame.copy()
            with self._lock:
                self._frame_buffer.append(self.current_frame)
            
            self.pixmap_signal.emit(frame)
            time.sleep(0.03)
        cap.release()

    def get_safe_frame(self):
        return self.current_frame

    def snapshot_frames_for_audit(self):
        with self._lock:
            return list(self._frame_buffer)

    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
# 3. INTERFACE "KALI EXPERT"
# =============================================================================

class CasinoExpert(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KALI-ROULETTE : EXPERT EDITION")
        self.setGeometry(50, 50, 1300, 850)

        # Style Global
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #f0f0f0; font-family: 'Segoe UI', sans-serif; }
            QFrame { background-color: #1e1e1e; border-radius: 12px; border: 1px solid #333; }
            QPushButton {
                background-color: #333; color: white; border-radius: 8px; padding: 10px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #444; border: 1px solid #666; }
            QPushButton:disabled { background-color: #222; color: #555; }
            QSpinBox { background: #222; color: #fff; border: 1px solid #444; padding: 5px; border-radius: 5px; }
            QRadioButton { color: #aaa; font-size: 14px; }
            QRadioButton::indicator:checked { background-color: #ffd700; border: 2px solid white; border-radius: 6px; }
        """)

        # Données
        self.bankroll = 1000.0
        self.current_bet_amount = 10.0
        self.video_file = "kalicasino.mp4"  # <--- VOTRE FICHIER VIDEO ICI

        if not os.path.exists(self.video_file):
            QMessageBox.critical(self, "Erreur", f"Vidéo manquante : {self.video_file}")
            sys.exit()

        self.init_ui()

        # Thread Vidéo
        self.thread = VideoThread(self.video_file)
        self.thread.pixmap_signal.connect(self.update_image)
        self.thread.start()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # --- COLONNE GAUCHE ---
        left = QVBoxLayout()

        # 1. Vidéo
        video_frame = QFrame()
        vf_layout = QVBoxLayout(video_frame)
        vf_layout.addWidget(QLabel("SOURCE D'ENTROPIE"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.img_lbl = QLabel()
        self.img_lbl.setMinimumSize(480, 270)
        self.img_lbl.setStyleSheet("background: black; border: 2px solid #444;")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vf_layout.addWidget(self.img_lbl)
        left.addWidget(video_frame)

        # 2. Audit
        audit_frame = QFrame()
        af_layout = QVBoxLayout(audit_frame)
        lbl_a = QLabel("AUDIT SCIENTIFIQUE")
        lbl_a.setStyleSheet("color: #ffa500; font-weight: bold; font-size: 16px; border: none;")
        af_layout.addWidget(lbl_a)

        h_audit = QHBoxLayout()
        h_audit.addWidget(QLabel("Nombre de tirages :"))
        self.spin_audit = QSpinBox()
        self.spin_audit.setRange(100, 2000000)
        self.spin_audit.setValue(50000)
        self.spin_audit.setSingleStep(1000)
        h_audit.addWidget(self.spin_audit)
        af_layout.addLayout(h_audit)

        self.btn_audit = QPushButton("📥 LANCER L'ANALYSE + SAUVEGARDER CSV")
        self.btn_audit.setStyleSheet("background-color: #0056b3; color: white;")
        self.btn_audit.clicked.connect(self.run_audit)
        af_layout.addWidget(self.btn_audit)

        self.prog_bar = QProgressBar()
        self.prog_bar.setStyleSheet("QProgressBar { border: none; background: #333; height: 8px; } QProgressBar::chunk { background: #00e5ff; }")
        af_layout.addWidget(self.prog_bar)

        # --- NOUVEAU : BOUTONS POUR GRAPHIQUES INDIVIDUELS ---
        self.layout_btns_graph = QHBoxLayout()
        
        self.btn_view_histo = QPushButton("📊 Histo")
        self.btn_view_reg = QPushButton("📈 Régularité")
        self.btn_view_esc = QPushButton("🪜 Escalier")

        # Désactivés au départ
        self.btn_view_histo.setEnabled(False)
        self.btn_view_reg.setEnabled(False)
        self.btn_view_esc.setEnabled(False)

        style_sub_btns = "background-color: #444; font-size: 12px; padding: 5px;"
        self.btn_view_histo.setStyleSheet(style_sub_btns)
        self.btn_view_reg.setStyleSheet(style_sub_btns)
        self.btn_view_esc.setStyleSheet(style_sub_btns)

        self.btn_view_histo.clicked.connect(self.show_histo_seul)
        self.btn_view_reg.clicked.connect(self.show_reg_seul)
        self.btn_view_esc.clicked.connect(self.show_escalier_seul)

        self.layout_btns_graph.addWidget(self.btn_view_histo)
        self.layout_btns_graph.addWidget(self.btn_view_reg)
        self.layout_btns_graph.addWidget(self.btn_view_esc)
        
        af_layout.addLayout(self.layout_btns_graph)
        # -----------------------------------------------------

        self.lbl_audit_status = QLabel("Prêt")
        self.lbl_audit_status.setStyleSheet("color: #888; font-style: italic; border: none;")
        self.lbl_audit_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        af_layout.addWidget(self.lbl_audit_status)

        left.addWidget(audit_frame)
        left.addStretch()
        layout.addLayout(left, 45)

        # --- COLONNE DROITE ---
        right = QVBoxLayout()

        # Bankroll
        bank_frame = QFrame()
        bank_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #ffd700;")
        bf_layout = QHBoxLayout(bank_frame)
        self.lbl_bank = QLabel(f"{self.bankroll:.0f} €")
        self.lbl_bank.setFont(QFont("Impact", 32))
        self.lbl_bank.setStyleSheet("color: #ffd700; border: none;")
        bf_layout.addWidget(QLabel("💰 CAPITAL :"))
        bf_layout.addWidget(self.lbl_bank)
        bf_layout.addStretch()
        right.addWidget(bank_frame)

        # Résultat
        res_frame = QFrame()
        rf_layout = QVBoxLayout(res_frame)
        self.lbl_res_num = QLabel("?")
        self.lbl_res_num.setFont(QFont("Impact", 90))
        self.lbl_res_num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_res_num.setFixedHeight(180)
        self.lbl_res_num.setStyleSheet("color: #333; background: #111; border-radius: 20px; border: 4px solid #333;")
        rf_layout.addWidget(self.lbl_res_num)

        self.lbl_res_txt = QLabel("FAITES VOS JEUX")
        self.lbl_res_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_res_txt.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_res_txt.setStyleSheet("border: none; color: #888;")
        rf_layout.addWidget(self.lbl_res_txt)
        right.addWidget(res_frame)

        # Paris
        bet_frame = QFrame()
        bet_layout = QVBoxLayout(bet_frame)
        self.bet_group = QButtonGroup()
        h_bets = QHBoxLayout()
        self.rb_red = QRadioButton("ROUGE (x2)")
        self.rb_red.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.rb_black = QRadioButton("NOIR (x2)")
        self.rb_black.setStyleSheet("color: white; font-weight: bold;")
        self.rb_green = QRadioButton("VERT (x36)")
        self.rb_green.setStyleSheet("color: #00cc00; font-weight: bold;")
        self.rb_none = QRadioButton("Juste regarder")
        self.rb_none.setChecked(True)

        self.bet_group.addButton(self.rb_red)
        self.bet_group.addButton(self.rb_black)
        self.bet_group.addButton(self.rb_green)
        self.bet_group.addButton(self.rb_none)

        h_bets.addWidget(self.rb_red)
        h_bets.addWidget(self.rb_black)
        h_bets.addWidget(self.rb_green)
        h_bets.addWidget(self.rb_none)
        bet_layout.addLayout(h_bets)

        self.btn_spin = QPushButton("🎲 LANCER LA BILLE (10€)")
        self.btn_spin.setFixedHeight(70)
        self.btn_spin.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4af37, stop:1 #a67c00);
                color: black; font-size: 20px; border-radius: 10px;
            }
            QPushButton:hover { background: #ffd700; }
        """)
        self.btn_spin.clicked.connect(self.start_spin_animation)
        bet_layout.addWidget(self.btn_spin)
        right.addWidget(bet_frame)

        # Historique
        hist_frame = QFrame()
        hf_layout = QHBoxLayout(hist_frame)
        hf_layout.addWidget(QLabel("HISTORIQUE :"))
        self.hist_layout = QHBoxLayout()
        self.hist_widgets = []
        hf_layout.addLayout(self.hist_layout)
        hf_layout.addStretch()
        right.addWidget(hist_frame)

        layout.addLayout(right, 55)

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qt_img.scaled(480, 270, Qt.AspectRatioMode.KeepAspectRatio)))

    # ==================== JEU ====================

    def start_spin_animation(self):
        if not self.rb_none.isChecked() and self.bankroll < self.current_bet_amount:
            QMessageBox.warning(self, "Fonds Insuffisants", "Vous êtes ruiné !")
            return
        self.btn_spin.setEnabled(False)
        self.lbl_res_txt.setText("Rien ne va plus...")
        self.anim_timer = QTimer()
        self.anim_steps = 0
        self.anim_timer.timeout.connect(self.anim_tick)
        self.anim_timer.start(50)

    def anim_tick(self):
        fake = random.randint(0, 36)
        c_hex, _ = RouletteLogic.get_info(fake)
        self.lbl_res_num.setText(str(fake))
        self.lbl_res_num.setStyleSheet(f"color: #555; background: {c_hex}; border-radius: 20px; border: 4px solid #555;")
        self.anim_steps += 1
        if self.anim_steps > 15:
            self.anim_timer.stop()
            self.finalize_spin()

    def finalize_spin(self):
        frame = self.thread.get_safe_frame()
        if frame is None:
            self.btn_spin.setEnabled(True)
            return
        res = EntropyEngine.get_spin(frame)
        num, hex_col, name = res['number'], res['hex'], res['name']
        self.lbl_res_num.setText(str(num))
        self.lbl_res_num.setStyleSheet(f"color: white; background: {hex_col}; border-radius: 20px; border: 6px solid white;")
        self.process_bet(num, name)
        self.add_visual_history(num, hex_col)
        self.btn_spin.setEnabled(True)

    def process_bet(self, number, color_name):
        bet_type = "NONE"
        if self.rb_red.isChecked(): bet_type = "ROUGE"
        elif self.rb_black.isChecked(): bet_type = "NOIR"
        elif self.rb_green.isChecked(): bet_type = "VERT"

        if bet_type == "NONE":
            self.lbl_res_txt.setText(f"RÉSULTAT : {color_name}")
            self.lbl_res_txt.setStyleSheet("color: white; border: none; font-size: 18px;")
            return

        self.bankroll -= self.current_bet_amount
        win = 0
        won = False
        if bet_type == "ROUGE" and "ROUGE" in color_name:
            win = self.current_bet_amount * 2
            won = True
        elif bet_type == "NOIR" and "NOIR" in color_name:
            win = self.current_bet_amount * 2
            won = True
        elif bet_type == "VERT" and number == 0:
            win = self.current_bet_amount * 36
            won = True

        if won:
            self.bankroll += win
            self.lbl_res_txt.setText(f"GAGNÉ ! (+{win}€)")
            self.lbl_res_txt.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 20px; border: none;")
        else:
            self.lbl_res_txt.setText("PERDU...")
            self.lbl_res_txt.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 20px; border: none;")
        self.lbl_bank.setText(f"{self.bankroll:.0f} €")

    def add_visual_history(self, num, col):
        lbl = QLabel(str(num))
        lbl.setFixedSize(36, 36)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"background: {col}; color: white; border-radius: 18px; font-weight: bold;")
        self.hist_layout.insertWidget(0, lbl)
        self.hist_widgets.insert(0, lbl)
        if len(self.hist_widgets) > 10:
            w = self.hist_widgets.pop()
            w.deleteLater()

    # ==================== ANALYSE ====================

    def run_audit(self):
        self.btn_audit.setEnabled(False)
        self.target_audit = self.spin_audit.value()
        self.audit_data = []
        self.prog_bar.setValue(0)
        self.lbl_audit_status.setText("Analyse en cours...")
        self.timer_audit = QTimer()
        self.timer_audit.timeout.connect(self.audit_step)
        self.timer_audit.start(1)

    def audit_step(self):
        frames = self.thread.snapshot_frames_for_audit()
        if not frames:
            f = self.thread.get_safe_frame()
            if f is None: return
            frames = [f]
        
        for _ in range(1000):
            frame = random.choice(frames)
            res = EntropyEngine.get_spin(frame)
            self.audit_data.append(res["number"])

        pct = int(len(self.audit_data) / self.target_audit * 100)
        self.prog_bar.setValue(min(100, pct))

        if len(self.audit_data) >= self.target_audit:
            self.timer_audit.stop()
            self.finalize_audit()

    def finalize_audit(self):
        try:
            filename = "historique_analyse.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Tirage N°", "Numéro Sorti", "Couleur"])
                for i, num in enumerate(self.audit_data):
                    _, color_name = RouletteLogic.get_info(num)
                    writer.writerow([i + 1, num, color_name])
            self.lbl_audit_status.setText(f"✅ Terminé ! Fichier '{filename}' créé.")
        except Exception as e:
            self.lbl_audit_status.setText("Erreur sauvegarde CSV")

        # Affichage du résumé
        self.show_graph()
        
        # Réactivation des boutons
        self.btn_audit.setEnabled(True)
        self.btn_view_histo.setEnabled(True)
        self.btn_view_reg.setEnabled(True)
        self.btn_view_esc.setEnabled(True)

    # ==================== GRAPHIQUES ====================

    def show_graph(self):
        """Affiche le résumé (3 courbes empilées)."""
        plt.close('all')
        counts = [self.audit_data.count(i) for i in range(37)]
        expected = len(self.audit_data) / 37
        chi2_score = sum([((c - expected) ** 2) / expected for c in counts])
        is_safe = chi2_score < 60.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11))
        plt.subplots_adjust(hspace=0.55)
        fig.patch.set_facecolor('#202020')
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#303030')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        colors = ['green'] + ['#D00000' if i in RouletteLogic.RED_NUMBERS else '#101010' for i in range(1, 37)]
        ax1.bar(range(37), counts, color=colors, edgecolor='gray')
        ax1.axhline(expected, color='cyan', linestyle='--')
        ax1.set_title("Histogramme")

        sorted_counts = sorted(counts)
        ax2.plot(range(37), sorted_counts, color='#bd93f9', linewidth=2, marker='o')
        ax2.axhline(expected, color='cyan', linestyle='--')
        ax2.set_title("Régularité")

        # Escalier miniature
        sorted_values = np.sort(np.array(self.audit_data, dtype=np.int16))
        n = len(sorted_values)

        max_points = 200_000
        step = max(1, n // max_points)

        y = sorted_values[::step]
        x = np.arange(0, n, step)[:len(y)]

        ax3.step(x, y, where="post", color="#f1c40f")
        ax3.set_title("Escalier (Aperçu, axe réel)")
        ax3.set_xlim(0, n)  # <-- important

        status = "CERTIFIÉ ÉQUITABLE" if is_safe else "ATTENTION BIAIS"
        col_s = "#00ff00" if is_safe else "red"
        fig.suptitle(f"{status}\nScore Chi2: {chi2_score:.2f}", fontsize=14, color=col_s, weight='bold')
        plt.show()

    def show_histo_seul(self):
        """Affiche uniquement l'Histogramme en grand."""
        if not self.audit_data: return
        
        counts = [self.audit_data.count(i) for i in range(37)]
        expected = len(self.audit_data) / 37
        
        plt.figure(figsize=(12, 7))
        plt.style.use('dark_background')
        
        colors = ['green'] + ['#D00000' if i in RouletteLogic.RED_NUMBERS else '#333' for i in range(1, 37)]
        plt.bar(range(37), counts, color=colors, edgecolor='white', alpha=0.8)
        plt.axhline(expected, color='cyan', linestyle='--', label='Moyenne théorique')
        
        plt.title(f"HISTOGRAMME DÉTAILLÉ ({len(self.audit_data)} tirages)", fontsize=16)
        plt.xlabel("Numéro")
        plt.ylabel("Sorties")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

    def show_reg_seul(self):
        """Affiche uniquement la courbe de régularité en grand."""
        if not self.audit_data: return

        counts = [self.audit_data.count(i) for i in range(37)]
        sorted_counts = sorted(counts)
        expected = len(self.audit_data) / 37

        plt.figure(figsize=(12, 7))
        plt.style.use('dark_background')
        
        plt.plot(range(37), sorted_counts, color='#bd93f9', linewidth=4, marker='o', markersize=8)
        plt.axhline(expected, color='cyan', linestyle='--')
        plt.fill_between(range(37), expected * 0.9, expected * 1.1, color='cyan', alpha=0.15, label="Zone de tolérance (±10%)")
        
        plt.title("ANALYSE DE RÉGULARITÉ (Biais)", fontsize=16)
        plt.xlabel("Numéros triés par fréquence (du moins sorti au plus sorti)")
        plt.ylabel("Nombre de sorties")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def show_escalier_seul(self):
        """Affiche la courbe Escalier seule (Style sombre + Jaune) avec axe X à l'échelle réelle."""
        if not self.audit_data:
            return

        sorted_values = np.sort(np.array(self.audit_data, dtype=np.int16))
        n = len(sorted_values)

        # Sous-échantillonnage pour la performance si nécessaire
        max_points = 200_000
        if n > max_points:
            step = max(1, n // max_points)
            sorted_values_plot = sorted_values[::step]

            # IMPORTANT : garder l'axe X à l'échelle réelle (0..n)
            x = np.arange(0, n, step)
            x = x[:len(sorted_values_plot)]  # sécurité taille identique
        else:
            sorted_values_plot = sorted_values
            x = np.arange(n)

        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')

        plt.step(x, sorted_values_plot, where="post", color='#FFC107', linewidth=2, label='Valeurs triées')

        plt.title(f"COURBE ESCALIER (Données brutes triées – 0..36) — N={n}", fontsize=15, pad=15)
        plt.xlabel("Index (dans la liste triée)")
        plt.ylabel("Valeur (0..36)")
        plt.ylim(-1, 37)

        # IMPORTANT : force l'affichage complet jusqu'à 1 000 000
        plt.xlim(0, n)

        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CasinoExpert()
    win.show()
    sys.exit(app.exec())