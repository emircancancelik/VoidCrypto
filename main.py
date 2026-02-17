import ccxt
import sys
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                               QMainWindow, QLineEdit, QVBoxLayout,
                               QLabel, QMenu, QMessageBox, QTextEdit, 
                               QFrame, QHBoxLayout)

#css 
DARK_STYLE = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', Arial;
}
QLineEdit {
    background-color: #1E1E1E;
    border: 2px solid #333333;
    border-radius: 8px;
    padding: 10px;
    color: white;
    font-size: 13px;
}
QLineEdit:focus {
    border: 2px solid #F3BA2F;
}
QPushButton {
    background-color: #F3BA2F;
    color: #000000;
    border-radius: 8px;
    font-weight: bold;
    font-size: 14px;
    padding: 12px;
}
QPushButton:hover {
    background-color: #FFD56B;
}
QPushButton:pressed {
    background-color: #C79820;
}
QTextEdit {
    background-color: #0A0A0A;
    border: 1px solid #333333;
    border-radius: 8px;
    color: #00FF00;
    font-family: 'Consolas', 'Courier New';
    font-size: 12px;
}
QLabel#Title {
    color: #F3BA2F;
    font-size: 22px;
    font-weight: bold;
}
"""

class BinanceApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoidCrypto v1.0")
        self.setFixedSize(480, 600)
        self.setStyleSheet(DARK_STYLE)

        # Ana Widget ve Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(25, 25, 25, 25)

        # --- Üst Başlık ---
        self.label_title = QLabel("VoidCrypto")
        self.label_title.setObjectName("Title")
        self.label_title.setAlignment(Qt.AlignCenter)
        
        self.status_label = QLabel("Bağlantı Bekleniyor...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888888; font-size: 11px;")

        # --- Giriş Alanları Çerçevesi ---
        self.setup_ui_elements()

        # --- Log Ekranı ---
        self.log_screen = QTextEdit()
        self.log_screen.setReadOnly(True)
        self.log_screen.setPlaceholderText("Sistem mesajları burada görünecek...")

        # Yerleştirme
        self.main_layout.addWidget(self.label_title)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.create_input_group("API KEY", self.api_key_input))
        self.main_layout.addWidget(self.create_input_group("SECRET KEY", self.secret_key_input))
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.btn_connect)
        self.main_layout.addWidget(QLabel("İşlem Logları:"))
        self.main_layout.addWidget(self.log_screen)

    def setup_ui_elements(self):
        # API Key Girişi
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Binance API Key...")
        
        # Secret Key Girişi
        self.secret_key_input = QLineEdit()
        self.secret_key_input.setPlaceholderText("Binance Secret Key...")
        self.secret_key_input.setEchoMode(QLineEdit.Password)

        # Bağlan Butonu
        self.btn_connect = QPushButton("BAĞLANTIYI TEST ET")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.clicked.connect(self.connect_binance)

    def create_input_group(self, label_text, widget):
        group_widget = QWidget()
        layout = QVBoxLayout(group_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-weight: bold; color: #F3BA2F; font-size: 11px;")
        
        layout.addWidget(lbl)
        layout.addWidget(widget)
        return group_widget

    def connect_binance(self):
        api = self.api_key_input.text().strip()
        sec = self.secret_key_input.text().strip()

        if not api or not sec:
            QMessageBox.critical(self, "Hata", "API bilgileri eksik!")
            return

        self.log_screen.clear()
        self.log_screen.append("<span style='color: white;'>[SİSTEM]</span> Bağlantı deneniyor...")
        self.status_label.setText("Bağlanıyor...")

        try:
            exchange = ccxt.binance({
                'apiKey': api,
                'secret': sec,
                'enableRateLimit': True
            })

            # Bakiye çekme testi
            balance = exchange.fetch_balance()
            
            self.log_screen.append("<span style='color: #00FF00;'>[BAŞARILI]</span> Binance bağlantısı doğrulandı.")
            self.log_screen.append("-" * 30)
            
            has_funds = False
            for asset, total in balance['total'].items():
                if total > 0:
                    self.log_screen.append(f"<b>{asset}</b>: {total}")
                    has_funds = True
            
            if not has_funds:
                self.log_screen.append("Cüzdanda varlık bulunamadı.")
            
            self.log_screen.append("-" * 30)
            self.status_label.setText("Bağlantı Aktif")
            self.status_label.setStyleSheet("color: #00FF00; font-size: 11px;")

        except Exception as e:
            self.log_screen.append(f"<span style='color: #FF0000;'>[HATA]</span> {str(e)}")
            self.status_label.setText("Bağlantı Hatası!")
            self.status_label.setStyleSheet("color: #FF0000; font-size: 11px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = BinanceApp()
    window.show()
    sys.exit(app.exec())