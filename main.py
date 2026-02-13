import ccxt
import sys

from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                               QMainWindow)
# crypto

class VoidCrypto:
    def __init__(self):   
        super().__init__()
        #self.crypto_data = VoidCrypto()
        #self.sembol = sembol  # bu sekilde istenilen kripto para bilgileri ekrana getirilebilir
    #BTC 
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        print("BTC/USDT")
        print(f"Last Price: {ticker['last']}")
        print(f"Buy: (Bid): {ticker['bid']}")  
        print(f"Sell: (Ask): {ticker['ask']}") 
        print(f"Max Value: {ticker['quoteVolume']}")

        print("====================================================")
    #GALA
        print("GALA/USDT")
        ticker = exchange.fetch_ticker('GALA/USDT')
        print(f"Last Price: {ticker['last']}")
        print(f"Buy: (Bid): {ticker['bid']}")  
        print(f"Sell: (Ask): {ticker['ask']}") 
        print(f"Max Value: {ticker['quoteVolume']}")

        print("====================================================")
    #ACH
        print("ACH/USDT")
        ticker = exchange.fetch_ticker('ACH/USDT')
        print(f"Last Price: {ticker['last']}")
        print(f"Buy: (Bid): {ticker['bid']}")  
        print(f"Sell: (Ask): {ticker['ask']}") 
        print(f"Max Value: {ticker['quoteVolume']}")

# ui

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoidCrypto")
        self.setFixedSize(QSize(400, 300))

        button = QPushButton("Press Me!")
        button.clicked.connect(self.the_button_was_clicked)
        button.clicked.connect(self.the_button_was_toggled)
        self.setCentralWidget(button) # pencereyi merkeze koyar

    def the_button_was_clicked(self):
        print("Clicked!")

    def the_button_was_toggled(self, checked):
        print("Checked?", checked)


# wallet

class KriptoCuzdan:
    pass
    def __init__(self, cüzdan_adi, bakiye):
        # Dışarıdan gelen verileri nesnenin içine 'mühürlüyoruz'
        self.name = cüzdan_adi 
        self.balance = bakiye

    def bilgileri_goster(self):
        # self sayesinde bu verilere her yerden ulaşabiliriz
        print(f"Cüzdan: {self.name}, Bakiye: {self.balance}")

# Nesneyi oluştururken verileri gönderiyoruz
cuzdan1 = KriptoCuzdan("Ana Cuzdan", 1500)
cuzdan1.bilgileri_goster()


app = QApplication(sys.argv)
VoidCrypto()
window = MainWindow()
window.show()

app.exec()

