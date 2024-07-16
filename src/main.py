from PyQt5.QtWidgets import QApplication
import sys
from frame.file_selector import FileSelector

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FileSelector()
    window.show()
    sys.exit(app.exec_())

# from PyQt5.QtWidgets import QApplication, QFileDialog
# import sys
# from frame.stl_viewer import STLViewer

# def main():
#     app = QApplication(sys.argv)
#     file_dialog = QFileDialog()
#     file_dialog.setNameFilter("STL files (*.stl)")
#     file_dialog.setFileMode(QFileDialog.ExistingFile)

#     if file_dialog.exec_():
#         stl_file = file_dialog.selectedFiles()[0]
#         window = STLViewer(stl_file)
#         window.show()
#         sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()
