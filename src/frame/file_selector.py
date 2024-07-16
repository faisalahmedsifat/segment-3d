from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QLineEdit, QMessageBox
import sys
from frame.segmentation_tool import SegmentationTool

class FileSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stl_file = None
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 200)
        self.setWindowTitle('Select STL File')

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.label = QLabel("Select an STL file to open:", self)
        layout.addWidget(self.label)

        self.file_path = QLineEdit(self)
        self.file_path.setReadOnly(True)
        layout.addWidget(self.file_path)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.browse_button)

        self.continue_button = QPushButton("Continue", self)
        self.continue_button.clicked.connect(self.continue_to_app)
        layout.addWidget(self.continue_button)

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("STL files (*.stl)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            self.stl_file = file_dialog.selectedFiles()[0]
            self.file_path.setText(self.stl_file)

    def continue_to_app(self):
        if self.stl_file:
            self.hide()
            self.stl_viewer = SegmentationTool(self.stl_file)
            self.stl_viewer.show()
        else:
            QMessageBox.warning(self, "No file selected", "Please select an STL file to continue.")

    def closeEvent(self, event):
        # self.temp_dir.cleanup()
        event.accept()