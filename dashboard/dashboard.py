import sys
import json
import redis
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QTextEdit, QWidget
from PySide6.QtCore import QThread, Signal, Qt

# --- ASENKRON REDIS DİNLEYİCİ ---
class RedisWorker(QThread):
    # Yeni bir sinyal/mesaj geldiğinde UI'ı tetiklemek için sinyaller
    message_received = Signal(str)
    consensus_received = Signal(dict)

    def run(self):
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            pubsub = r.pubsub()
            pubsub.subscribe(['void_master_decision', 'void_agent_signals'])
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    if message['channel'] == 'void_agent_signals':
                        self.message_received.emit(message['data'])
                    elif message['channel'] == 'void_master_decision':
                        self.consensus_received.emit(json.loads(message['data']))
        except Exception as e:
            self.message_received.emit(f"Hata: {str(e)}")

# --- ANA PENCERE ---
class VoidCryptoTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoidCrypto Orchestrator Terminal v1.0")
        self.setMinimumSize(800, 500)
        self.setStyleSheet("background-color: #121212; color: #E0E0E0;")

        # Layout Düzeni
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Altyapı Bekleniyor...")
        self.status_label.setStyleSheet("font-size: 18px; color: #00FF00; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #1E1E1E; border: 1px solid #333; font-family: 'Courier New';")
        layout.addWidget(self.log_area)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Worker Başlatma
        self.worker = RedisWorker()
        self.worker.message_received.connect(self.update_logs)
        self.worker.consensus_received.connect(self.update_consensus)
        self.worker.start()

    def update_logs(self, msg):
        self.log_area.append(f"> {msg}")

    def update_consensus(self, data):
        action = data.get("action", "WAIT")
        conf = data.get("confidence", 0)
        self.status_label.setText(f"KARAR: {action} | GÜVEN: %{conf}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoidCryptoTerminal()
    window.show()
    sys.exit(app.exec())