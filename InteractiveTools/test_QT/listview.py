import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QVBoxLayout, QWidget, QAbstractItemView, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("List View with Checkboxes")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.list_view = QListView(self)
        self.list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing

        # Set up the model
        self.model = QStandardItemModel(self.list_view)

        # Add items to the model
        for i in range(10):
            item = QStandardItem(f"Item {i+1} is here")
            item.setCheckable(True)
            self.model.appendRow(item)

        self.list_view.setModel(self.model)
        self.layout.addWidget(self.list_view)

        # Add a button to trigger the word color change
        self.change_button = QPushButton("Change 'here' color to red", self)
        self.change_button.clicked.connect(self.change_word_color)
        self.layout.addWidget(self.change_button)

    def change_word_color(self):
        row = 2  # Specify the row you want to modify (0-based index)
        target_word = "here"
        color = "red"
        item = self.model.item(row)

        if item:
            text = item.text()
            # Replace the target word with the colored word
            updated_text = text.replace(target_word, f'<span style="color:{color};">{target_word}</span>')
            item.setText(updated_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
