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
)

class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        #self.third = QComboBox(self)      
        self.fourth = QComboBox(self)
        self.fifth = QComboBox(self)
        
        #third_list = ["4-0 / 0.4", "4-0 / 0.6", "5-0 / 0.6", "5-0 / 1.0", "8-0 / 1.0", "8-0 / 1.6", "10-0 / 1.6", "10-0 / 2.5"]
        fourth_list = ["Inhouse", '2 Week', '4 Week']
        fifth_list = [str(x) for x in range(1, 4)]  
        
        #self.third.addItems(third_list)
        self.fourth.addItems(fourth_list)
        self.fifth.addItems(fifth_list)
        #self.third.setEditable(True)
        self.fourth.setEditable(True)
        self.fifth.setEditable(True)
        
        self.video_path = QLabel(self)
        self.video_path.setText("No video selected")
        self.browseBtn = QPushButton(self)
        self.browseBtn.setText("Browse Video")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self);
        self.browseBtn.clicked.connect(self.get_video)
        layout = QFormLayout(self)
        layout.addRow("", self.browseBtn)
        layout.addRow("Selected Video Path: ", self.video_path)
        layout.addRow("Name: ", self.first)
        layout.addRow("Email: ", self.second)
        #layout.addRow("Thread / Magnification: ", self.third)
        layout.addRow("Program: ", self.fourth)
        layout.addRow("Iteration Number: ", self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def get_video(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open file')
        if len(file_name) != 0:
            self.video_path.setText(file_name[0])

    def showdialog(self, text1, text2):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def getInputs(self):
        if self.first.text() != None and self.first.text() != "" and self.second.text() != None and self.second.text() != "":
            return {
                "name": self.first.text(),
                "email": self.second.text(),
                #"recording_type": self.third.currentText(),
                "program": self.fourth.currentText(),
                "iteration": self.fifth.currentText(),
                "video_path": self.video_path.text()
            }
        else:
            self.showdialog(text1="No name or email found", text2="Please enter a valid name and email to proceed")
            return None