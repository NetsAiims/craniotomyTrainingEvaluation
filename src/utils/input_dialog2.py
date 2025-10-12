import base64
import os
from PyQt5.QtWidgets import (
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QFormLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QMessageBox,
    QPlainTextEdit
)

class InputDialog2(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QComboBox(self)
        self.second = QLineEdit(self)
        self.third = QPlainTextEdit(self)
        
        # UI text changed back to "Report"
        self.browseBtn = QPushButton("Browse Report...", self)
        self.path_label = QLabel("No report selected", self)

        first_list = ["Anonymous Feedback", "Non Anonymous Feedback"]        
        
        self.first.addItems(first_list)
        self.first.setEditable(False)
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        layout.addRow("Feedback Type: ", self.first)
        layout.addRow("Name: ", self.second)
        layout.addRow("Feedback: ", self.third)        
        layout.addRow("", self.browseBtn)
        # UI text changed back to "Report"
        layout.addRow("Selected Report Path: ", self.path_label)
        layout.addWidget(buttonBox)
        
        self.browseBtn.clicked.connect(self.get_report_file) # Renamed for clarity
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.first.currentTextChanged.connect(self.changed_feedback_type)
        self.second.setEnabled(False)

    def get_report_file(self):
        """Opens a file dialog to select a PDF file."""
        # Filter changed to look for PDF files
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Select a PDF Report', 
            '', 
            "PDF Files (*.pdf)"
        )
        if file_path:
            self.path_label.setText(file_path)

    def changed_feedback_type(self, value):
        if value == "Anonymous Feedback":
            self.second.setEnabled(False)
        else:
            self.second.setEnabled(True)
            
    def showdialog(self, text1, text2):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def submit_feedback(self):
        """
        Reads the selected PDF file, encodes it to base64, and returns it
        along with the other feedback details.
        """
        if not self.third.toPlainText().strip():
            self.showdialog(text1='Please enter feedback', text2="Feedback cannot be empty.")
            return None

        file_path = self.path_label.text()
        
        base_output = {
            "feedback_type": self.first.currentText(),
            "name": self.second.text(),
            "feedback": self.third.toPlainText()
        }

        if file_path == "No report selected":
            return base_output
        else:
            try:
                with open(file_path, "rb") as report_file:
                    encoded_bytes = base64.b64encode(report_file.read())
                
                encoded_string = encoded_bytes.decode('utf-8')
                file_name = os.path.basename(file_path)

                base_output["file_name"] = file_name
                base_output["file_encoded_string"] = encoded_string
                return base_output

            except Exception as e:
                self.showdialog("File Error", f"Could not read the PDF file: {e}")
                return None