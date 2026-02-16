import ccxt
import sys

from random import choice
from PySide6.QtGui import QAction
from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                               QMainWindow, QLineEdit, QVBoxLayout,
                               QLabel, QMenu)
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
window_titles = [
    "My App",
    "My App",
    "Still My App",
    "Still My App",
    "What on earth",
    "What on earth",
    "This is surprising",
    "This is surprising",
    "Something went wrong",
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)
        #mouseevent
        self.setMouseTracking(True)
        self.label = QLabel("Click in this window")
        self.label.setMouseTracking(True)
        self.setCentralWidget(self.label)
        #pencere
        self.setWindowTitle("VoidCrypto")
        self.setFixedSize(QSize(400, 300))
        self.n_times_clicked = 0
        self.label = QLabel("Click in this window")

        #
        self.setCentralWidget(self.label)

        button = QPushButton("Press Me!")
        button.clicked.connect(self.the_button_was_clicked)
        button.clicked.connect(self.the_button_was_toggled)
        self.setCentralWidget(button) # pencereyi merkeze koyar

        self.windowTitleChanged.connect(self.the_window_title_changed)
     #Label
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)  
        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label) 
    def the_button_was_clicked(self):
        print("Clicked.")
        new_window_title = choice(window_titles)
        print("Setting title:  %s" % new_window_title)
        self.setWindowTitle(new_window_title)

    def the_window_title_changed(self, window_title):
        print("Window title changed: %s" % window_title)
        if window_title == "Something went wrong":
            self.button.setDisabled(True)

    def the_button_was_toggled(self, checked):
        print("Checked?", checked)

    def the_button_was_released(self):
        self.button_is_checked = self.button.isChecked()

    #mouseEvent
    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, e):
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        self.label.setText("mouseDoubleClickEvent")
    
    #istenilen fonksiyonun özelliğini alır ama orijinalliğini korur.
    #sonradan içeriye yazdığımız kodu da aldığımız kodu çalıştırır.
    def mousePressEvent(self, event):
        print("Mouse pressed!")
        super().mousePressEvent(event)    

    #contextMenu
    def on_context_menu(self, pos):
        context = QMenu(self)
        context.addAction(QAction("test 1", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(self.mapToGlobal(pos))

    def contextMenuEvent(self, e):
        context = QMenu(self)
        context.addAction(QAction("test 1", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(e.globalPos())    

        print(self.button_is_checked)
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

