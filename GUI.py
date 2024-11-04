from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QMessageBox, QVBoxLayout, QWidget

import pickle

class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Érzelemfelismerő")
        self.setGeometry(100, 100, 400, 300)


        self.label = QLabel("Írd be a szöveget:")
        self.text_edit = QTextEdit()
        self.classify_button = QPushButton("Osztályozás")
        self.classify_button.clicked.connect(self.classify_text)


        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.classify_button)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)




    #model megnyitasa def load_model(self):


    def classify_text(self):
        user_input = self.text_edit.toPlainText().strip()
        if not user_input:
            QMessageBox.warning(self, "Hiba", "Kérlek, adj meg egy szöveget.")
            return


        sentiment = self.model.predict([user_input])[0]
        QMessageBox.information(self, "Eredmény", f"Az érzelem: {sentiment}")



