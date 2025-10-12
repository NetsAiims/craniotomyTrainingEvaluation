import os
import sys
import cv2
from PIL import Image
import numpy as np
from src.modelling.model import Model
import threading
import time
import numpy as np
import warnings
import shutil
import requests
import ffmpeg

import json
import datetime
import smtplib
import base64
import matplotlib.pyplot as plt

from src.utils.videothread import VideoThread
from src.utils.input_dialog import InputDialog
from src.utils.pdf_writer import PdfWrite
from src.utils.annotator import Annotator
from src.utils.input_dialog2 import InputDialog2
from src.utils.input_dialog import InputDialog
from credentials import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, pyqtSlot, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QFileDialog
)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import sys
import psutil
import platform
import faulthandler
faulthandler.enable()

if "Windows" in platform.uname().system:
    from pygrabber.dshow_graph import FilterGraph

global capture
global img
global fileName
global gradcamFileName
global preEvaluateGradcam

preEvaluateGradcam = 0
image = None
originalFileName = None
fileName = None
originalImg = None
img = None
capture = None
warnings.filterwarnings("ignore")

class Application(QMainWindow):
    @property
    def runInit(self):
        return self._runInit

    @runInit.setter
    def runInit(self, value):
        self._runInit = value
        if value == True:
            self.init_complete()

    def __init__(self):
        global capture
        super().__init__()
        self._runInit = False
        self.placeholderText = [
            "Final score:"
        ]
        self.scoreText = [
            "__ / 10" for _ in range(6)
        ]
    
        self.mainWidget = QWidget()
        self.display_width = int(app.primaryScreen().size().width() * 0.4)
        self.display_height = int(app.primaryScreen().size().height() * 0.3)

        self.mainLayout = QVBoxLayout()
        self.set_top_header()
        self.set_tabs()
        self.setupBottomLoading()

        self.mainLayout.addLayout(self.topHeader)
        self.mainLayout.addWidget(self.tabs)
        self.mainLayout.addLayout(self.bottomBox)
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)
        self.tabs.currentChanged.connect(self.onChange)
        self.photobbox = None
        self.camerabbox = None
        self.mainWidget.setFixedSize(app.primaryScreen().size().width(), app.primaryScreen().size().height())

        self.model = None
        self.loading_thread = threading.Thread(target=self.get_model)
        self.loading_thread.start()
    
    def get_capture_devices(self):
        '''Detects and returns a list of available video capture devices connected'''
        index = 0
        arr = ["No source selected"]
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(f"Camera Input {index + 1}")
            cap.release()
            index += 1
        return arr

    def setupBottomLoading(self):
        self.bottomBox = QHBoxLayout()
        self.powerOff = QPushButton("Exit")
        self.bottomBox.addWidget(self.powerOff)
        self.powerOff.clicked.connect(self.shutdown)
        if "Windows" in platform.uname().system:
            graph = FilterGraph()
            self.input_devices = ["No source selected"] + graph.get_input_devices()
        elif "Darwin" in platform.uname().system or "Linux" in platform.uname().system:
            self.input_devices = self.get_capture_devices()
        self.sourceBtn = QComboBox()
        self.sourceBtn.addItems(self.input_devices)
        self.sourceBtn.activated.connect(self.changeCameraSource)
        self.bottomBox.addWidget(self.sourceBtn)
        for _ in range(2):
            self.bottomBox.addStretch()
        self.infoLbl = QLabel("Loading Model")
        self.bottomBox.addWidget(self.infoLbl)
        self.progress = QProgressBar()
        self.progress.hide()
        self.bottomBox.addWidget(self.progress)
        self.progress.setValue(0)

    def shutdown(self):
        print("Shutting down system")
        self.close()

    def changeCameraSource(self):
        global capture
        i = self.sourceBtn.currentIndex()
        if self.input_devices[i] == "No source selected":
            capture = None
            self.th.stop()
            self.th.stop_recording()
            self.th.quit()
            self.showdialog(text1="Invalid option", text2="Source is already selected.")
            self.tabs.setCurrentIndex(0)
            self.cameraLbl.setPixmap(QPixmap("src/images/placeholder.jpg").scaled(self.display_width, self.display_height))
            return
        if len(self.input_devices) > 1:
            self.sourceBtn.model().item(0).setEnabled(False)
        else:
            self.sourceBtn.model().item(0).setEnabled(True)
        
        try:
            if capture is not None:
                self.th.stop()
                self.th.stop_recording()
                self.th.quit()
                capture = None
            capture = cv2.VideoCapture(i - 1)
        except:
            self.showdialog(text1="Cant obtain selected device", text2="Cannot obtain selected device. Please try again")
            capture = None
            return
        time.sleep(1)
        self.tabs.setCurrentIndex(0)

    def updateProgress(self, value):
        self.progress.setValue(value)

    def onChange(self, i):
        '''Triggered when the input camera is changed from the GUI'''
        global img
        global capture
        if i == 1:
            self.th = VideoThread(capture)
            self.th.change_pixmap_signal.connect(self.update_image)
            self.th.change_timestamp_signal.connect(self.update_timestamp)
            self.th.start(QThread.HighestPriority)
        
    def get_model(self):            
            self.model = Model(start_callback=lambda: print("Starting loading model"), end_callback=lambda: self.init_complete(), progress_callback=lambda x: self.updateProgress(100))
       
    def init_complete(self):
        time.sleep(2)
        self.infoLbl.setText("Done loading model")

    def changeInit(self):
        self.runInit = True

    def set_top_header(self):
        self.topHeader = QHBoxLayout()
        self.topHeader.addStretch()
        self.topLabel = QLabel()
        self.topLabel.setPixmap(QPixmap("src/images/iitd.png").scaled(100, 100, Qt.KeepAspectRatio))
        self.topHeader.addWidget(self.topLabel)
        self.label2 = QLabel("Micro Drilling Evaluation System")
        self.topHeader.addWidget(self.label2)
        self.label2.setFont(QFont("Arial", 30))
        self.label3 = QLabel()
        self.label3.setPixmap(QPixmap("src/images/aiims.png").scaled(100, 100, Qt.KeepAspectRatio))
        self.topHeader.addWidget(self.label3)
        self.topHeader.addStretch()
        self.topHeader.setSpacing(30)
        self.topHeader.setContentsMargins(20, 0, 20, 0)

    def set_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 16))
        self.tabs.setStyleSheet("QTabBar::tab { height: 35px; width: 100px;}")
        self.tabs.addTab(self.photoTabUI(), "Photo")
        self.tabs.addTab(self.cameraTabUI(), "Camera")
    
    def photoTabUI(self):
        """Create the Photo page UI."""
        self.photoTab = QWidget()
        self.mainPhotoLayout = QHBoxLayout()
        self.leftPhotoLayout = QVBoxLayout()

        self.selectPhotoBtn = QPushButton("Select Photo")
        self.selectPhotoBtn.setFont(QFont("Arial", 16))
        self.selectPhotoBtn.setStyleSheet("""
            QPushButton: {
                border-radius: 10px;
                border-color: black;
                border-width: 2px;
            }
        """)
        self.selectPhotoBtn.clicked.connect(self.open_file)
        self.selectPhotoBtn.setFixedHeight(40)

        self.selectedPhotoLbl = QLabel()
        
        self.selectedPhotoLbl.setPixmap(QPixmap("src/images/placeholder.jpg").scaled(self.display_width, self.display_height))
        self.selectedPhotoLayout = QHBoxLayout()
        self.selectedPhotoLayout.addStretch()
        self.selectedPhotoLayout.addWidget(self.selectedPhotoLbl)
        self.selectedPhotoLayout.addStretch()

        self.selectedPhotoLbl2 = QLabel()
        self.selectedPhotoLbl2.setPixmap(QPixmap("src/images/placeholder.jpg").scaled(self.display_width, self.display_height))
        self.selectedPhotoLayout2 = QHBoxLayout()
        self.selectedPhotoLayout2.addStretch()
        self.selectedPhotoLayout2.addWidget(self.selectedPhotoLbl2)
        self.selectedPhotoLayout2.addStretch()
        
        self.evaluatePhotoBtn = QPushButton("Evaluate")        
        self.evaluatePhotoBtn.clicked.connect(self.evaluate)
        self.evaluatePhotoBtn.setFont(QFont("Arial", 16))
        self.evaluatePhotoBtn.setFixedHeight(40)
        self.leftPhotoLayout.addWidget(self.selectPhotoBtn)
        self.leftPhotoLayout.addLayout(self.selectedPhotoLayout)
        self.leftPhotoLayout.addLayout(self.selectedPhotoLayout2)
        self.leftPhotoLayout.addWidget(self.evaluatePhotoBtn)

        self.resultPhotoBox = QHBoxLayout()

        self.GradcamPhotobutton = QPushButton("Grad Cam")
        self.GradcamPhotobutton.setFont(QFont("Arial", 16))
        self.GradcamPhotobutton.clicked.connect(self.gradcam)
        self.GradcamPhotobutton.setFixedHeight(40)

        self.satisfiedPhotoBtn = QPushButton("Satisfied")
        self.satisfiedPhotoBtn.setFont(QFont("Arial", 16))
        self.satisfiedPhotoBtn.clicked.connect(self.satisfied)
        self.satisfiedPhotoBtn.setFixedHeight(40)
        self.notSatisfiedPhotoBtn = QPushButton("Not Satisfied")
        self.notSatisfiedPhotoBtn.setFont(QFont("Arial", 16))
        self.notSatisfiedPhotoBtn.clicked.connect(self.unsatisfied)
        self.notSatisfiedPhotoBtn.setFixedHeight(40)
        self.photoImageSavedStatus = QLabel()
        self.photoImageSavedStatus.setAlignment(Qt.AlignCenter)
        self.photoImageSavedStatus.hide()

        self.photoGenerateBtn = QPushButton("Generate Report")
        self.photoFeedbackBtn = QPushButton("Submit Feedback")
        self.photoGenerateBtn.setFont(QFont("Arial", 16))
        self.photoGenerateBtn.clicked.connect(self.showInputDialog)
        self.photoGenerateBtn.setFixedHeight(40)
        self.photoFeedbackBtn.setFont(QFont("Arial", 16))
        self.photoFeedbackBtn.clicked.connect(self.feedback)
        self.photoFeedbackBtn.setFixedHeight(40)

        self.photoExtraButtonsLayout = QHBoxLayout()
        self.photoExtraButtonsLayout.addWidget(self.photoGenerateBtn)
        self.photoExtraButtonsLayout.addWidget(self.photoFeedbackBtn)

        self.scorePhotoField = QLineEdit()
        self.scorePhotoField.setFixedWidth(100)
        self.scorePhotoField.setFixedHeight(self.notSatisfiedPhotoBtn.geometry().size().height())

        self.resultPhotoBox.addWidget(self.satisfiedPhotoBtn)
        self.resultPhotoBox.addWidget(self.notSatisfiedPhotoBtn)
        self.resultPhotoBox.addWidget(self.photoImageSavedStatus)
        self.resultPhotoBox.addWidget(self.scorePhotoField)
        self.resultPhotoBox.addWidget(self.GradcamPhotobutton)
        
        self.rightPhotoLayout, self.rightPhotoWidgets, self.rightPhotoLabels, self.photoOverrideButton, self.photoWrongLbl = self.setupSelectLabels(num_labels=6)
        self.photoOverrideButton.clicked.connect(self.overrideEval)
        self.rightPhotoLayout.setContentsMargins(20, 100, 20, 100)
        
        self.rightPhotoLayout.addLayout(self.photoExtraButtonsLayout)       
        self.rightPhotoLayout.addLayout(self.resultPhotoBox)

        self.mainPhotoLayout.addLayout(self.leftPhotoLayout, 50)
        self.mainPhotoLayout.addLayout(self.rightPhotoLayout, 50)
        self.mainPhotoLayout.setSpacing(40)
        self.photoTab.setLayout(self.mainPhotoLayout)
        return self.photoTab    
    
    
    def feedback(self):
        msg = InputDialog2(self)
        if not msg.exec():
            return

        outputs = msg.submit_feedback()
        if outputs is None:
            return
        sender_email = report_sender_email
        sender_password = report_sender_password
        recipient_email = "netsaiims@gmail.com"

        if not sender_email or not sender_password:
            self.showdialog(
                text1="Configuration Error"
            )
            return

        if outputs["feedback_type"] == "Anonymous Feedback":
            title = "Anonymous Feedback for Drilling Evaluation System"
        else:
            title = f"Feedback from {outputs.get('name', 'N/A')} for Drilling Evaluation System"
       
        body = f"""
        <html>
        <body>
            <h2>Feedback Report</h2>
            <p>This is a feedback for the Drilling Effectiveness Evaluation System deployed at AIIMS, New Delhi.</p>
            <hr>
            <h3>Details:</h3>
            <ul>
                <li><b>Feedback Type:</b> {outputs.get("feedback_type", "N/A")}</li>
                <li><b>Name:</b> {outputs.get('name') if outputs.get('name') else 'Name not provided'}</li>
            </ul>
            <h3>Feedback Message:</h3>
            <p style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
                {outputs.get("feedback", "No feedback message provided.")}
            </p>
        </body>
        </html>
        """

        try:
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = title
            message.attach(MIMEText(body, "html"))

            if "file_encoded_string" in outputs and "file_name" in outputs:
                try:
                    file_data = base64.b64decode(outputs['file_encoded_string'])
                    attachment = MIMEApplication(file_data, _subtype="jpeg") 
                    attachment.add_header(
                        "Content-Disposition",
                        f"attachment; filename={outputs['file_name']}",
                    )
                    message.attach(attachment)
                    print(f"Attached file: {outputs['file_name']}")
                except Exception as e:
                    print(f"Error attaching file: {e}")
                    pass

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(message)
                print("Feedback email sent successfully!")
                self.showdialog(
                    text1="Feedback Sent",
                    text2="Your feedback has been successfully sent via email."
                )

        except smtplib.SMTPAuthenticationError:
            print("Authentication error. Check email/password or App Password.")
            self.showdialog(
                text1="Authentication Error",
                text2="Could not send feedback. Please check your email credentials and ensure you're using a Google App Password."
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            self.showdialog(
                text1="Error Sending Feedback",
                text2=f"An unexpected error occurred: {e}"
            )

    def satisfied(self):
        global originalFileName
        global originalImg
        if self.tabs.currentIndex() == 0:
            self.photoImageSavedStatus.setText("Image Saved")
        else:
            self.cameraImageSavedStatus.setText("Image Saved")
        index = len(os.listdir("data/Images"))
        try:
            if self.tabs.currentIndex() == 0:
                shutil.copy2(originalFileName, f"data/Images/{index}.jpg")
                x, y, w, h = self.photobbox
                info = {
                    "file_name": f"{index}.jpg",
                    "score": int(self.rightPhotoLabels[-1][-1].text().split(" / ")[0]),
                    "is_satisfied": True,
                    "bbox": {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    }
                }
                with open(f"data/Annotations/{index}.json", "w") as f:
                    json.dump(info, f)
                self.photoImageSavedStatus.setText("Image saved")
            else:
                cv2.imwrite(f"data/Images/{index}.jpg", originalImg)
                x, y, w, h = self.camerabbox
                info = {
                    "file_name": f"{index}.jpg",
                    "score": int(self.rightCameraLabels[-1][-1].text().split(" / ")[0]),
                    "is_satisfied": True,
                    "bbox": {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    }
                }
                with open(f"data/Annotations/{index}.json", "w") as f:
                    json.dump(info, f)
                self.cameraImageSavedStatus.setText("Image saved")
        except:
            self.showdialog(text1="Some error occured",
                            text2="Please try again later")
            return

    def unsatisfied(self):
        global originalFileName
        global originalImg
        index = len(os.listdir("data/Images"))
        if self.tabs.currentIndex() == 0:
            try:
                score = int(
                    self.rightPhotoLabels[-1][-1].text().split(" / ")[0])
                updated_score = int(self.scorePhotoField.text())
                if updated_score > 10 or updated_score < 0:
                    self.showdialog(
                        text1="Enter a valid score", text2="Please enter a score between range of 0 and 10")
                    return
                else:
                    shutil.copy2(originalFileName, f"data/Images/{index}.jpg")
                    x, y, w, h = self.photobbox
                    info = {
                        "file_name": f"{index}.jpg",
                        "score": score,
                        "is_satisfied": False,
                        "updated_score": updated_score,
                        "bbox": {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h
                        }
                    }
                    with open(f"data/Annotations/{index}.json", "w") as f:
                        json.dump(info, f)
                    self.photoImageSavedStatus.setText("Image saved")
            except:
                self.showdialog(
                    text1="No score entered", text2="Please enter a valid score you feel the image should have got.")
                return
        else:
            try:
                score = int(
                    self.rightCameraLabels[-1][-1].text().split(" / ")[0])
                updated_score = int(self.scoreCameraField.text().strip())
                if updated_score > 10 or updated_score < 0:
                    self.showdialog(
                        text1="Enter a valid score", text2="Please enter a score between range of 0 and 10")
                    return
                else:
                    cv2.imwrite(f"data/Images/{index}.jpg", originalImg)
                    x, y, w, h = self.camerabbox
                    info = {
                        "file_name": f"{index}.jpg",
                        "score": score,
                        "is_satisfied": False,
                        "updated_score": updated_score,
                        "bbox": {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h
                        }
                    }
                    with open(f"data/Annotations/{index}.json", "w") as f:
                        json.dump(info, f)
                    self.cameraImageSavedStatus.setText("Image saved")
            except:
                self.showdialog(
                    text1="No score entered", text2="Please enter a valid score you feel the image should have got.")
                return

    def overrideEval(self):
        global img
        global fileName
        if self.tabs.currentIndex() == 0:
            if fileName is None:
                print("None for index 0")
            else:
                image = cv2.imread(fileName)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                scores = self.model.override(image)
                self.scoreText = [f"{score} / 10" for score in scores]
                self.setRightText(self.rightPhotoLabels)
                self.photoImageSavedStatus.setText("Image not saved")
                self.photoImageSavedStatus.show()
        else:
            if img is None:
                print("None for index 1")
            else:
                self.setRightText(self.rightCameraLabels)
                scores = self.model.override(img)
                self.scoreText = [f"{score} / 10" for score in scores]
                self.setRightText(self.rightCameraLabels)
                self.cameraImageSavedStatus.setText("Image not saved")
                self.cameraImageSavedStatus.show()

    def evaluate(self):
        global img
        global fileName
        global gradcamFileName
        global preEvaluateGradcam

        preEvaluateGradcam = 1 

        print("Evaluating")
        self.progress.show()
        self.infoLbl.setText("Evaluating")
        if self.tabs.currentIndex() == 0:
            self.scoreText = ["__ / 10" for _ in range(6)]
            self.setRightText(self.rightPhotoLabels)
            if fileName is None:
                print("None for index 0")
                return
            else:
                image = cv2.imread(fileName)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                scores = self.model.evaluate(image)
                gradcamFileName = "data/temp/gradcam_static.jpg"

                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    gradcam_result = self.model.visualize(image)
                    Image.fromarray(np.array(gradcam_result)).save(gradcamFileName)
                    print(f"GradCAM saved to: {gradcamFileName}")
                except Exception as e:
                    print(f"Error saving GradCAM: {e}")
                    self.showdialog(text1="GradCAM Error", text2="Could not generate GradCAM visualization")
                    return
               
                self.scoreText = [f"{score} / 10" for score in scores]
                self.setRightText(self.rightPhotoLabels)
                self.photoImageSavedStatus.setText("Image not saved")
                self.photoImageSavedStatus.show()
        else:
            self.scoreText = ["__ / 10" for _ in range(6)]
            #self.scoreText = [" __ / 10"]
            self.setRightText(self.rightCameraLabels)
            if img is None:
                print("None for index 1")
                return
            else:
                self.setRightText(self.rightCameraLabels)
                scores = self.model.evaluate(img)               
                gradcamFileName = "data/temp/gradcam_camera.jpg"

                try:
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gradcam_result = self.model.visualize(image_rgb)
                    Image.fromarray(np.array(gradcam_result)).save(gradcamFileName)
                    print(f"GradCAM saved to: {gradcamFileName}")
                except Exception as e:
                    print(f"Error saving GradCAM: {e}")
                    self.showdialog(text1="GradCAM Error", text2="Could not generate GradCAM visualization")
                    return
                
                self.scoreText = [f"{score} / 10" for score in scores]
                self.setRightText(self.rightCameraLabels)
                
                self.cameraImageSavedStatus.setText("Image not saved")
                self.cameraImageSavedStatus.show()

    def setupSelectLabels(self, num_labels, cameraSection=False):
        self.selectedLayout = QVBoxLayout()
    
        if cameraSection:
            self.recordingLayout = QHBoxLayout()
            for _ in range(3):
                self.recordingLayout.addStretch()
            self.recordingStatus = QLabel()
            self.timeStatus = QLabel()
            self.timeStatus.setAlignment(Qt.AlignCenter)
            self.timeStatus.hide()
            self.recordingLayout.addWidget(self.timeStatus)
            self.recordingStatus.setAlignment(Qt.AlignCenter)
            self.recordingStatus.setText("Not Recording")
            self.recordingStatus.setObjectName("red")
            self.recordingStatus.setFixedHeight(30)
            self.recordingStatus.setFixedWidth(200)
            self.recordingStatus.setStyleSheet("""
                QLabel#red{
                    background-color: red;
                    border-radius: 10px;        
                    border-width: 2px;
                    border-color: #000000;
                    border-style: solid;
                }
            """)
            self.recordingStatus.hide()
            self.recordingLayout.addWidget(self.recordingStatus)
            self.selectedLayout.addLayout(self.recordingLayout)

        self.hbox = QHBoxLayout()
        self.overrideButton = QPushButton("Override")
        self.wrongImg = QLabel("Wrong Image!")
        self.overrideButton.setFont(QFont("Arial", 16))
        self.wrongImg.setFont(QFont("Arial", 16))
        self.wrongImg.setStyleSheet("""
            color: "red";
        """)
        self.overrideButton.hide()
        self.wrongImg.hide()
        self.hbox.addWidget(self.wrongImg)
        self.hbox.addWidget(self.overrideButton)
        
        self.selectedLayout.addLayout(self.hbox)

        self.selectedWidgets = []
        self.selectedLayout.setSpacing(10)
        self.selectImgRightLabels = [
            [
            QLabel("") for _ in range(2)
            ] 
        ]
        self.selectedLayout.addStretch()
        for index1, element in enumerate(self.selectImgRightLabels):
            qw = QWidget()
            qw.setStyleSheet(
                """
                QWidget {
                    background-color: white;
                    border-radius: 10px;        
                    border-width: 2px;
                    border-color: #000000;
                    border-style: solid;
                }
                QLabel {
                    border-width: 0px;
                }
                """
            )
            if index1 == len(self.selectImgRightLabels) - 1:
                self.selectedLayout.addSpacing(40)
            self.horizontalLabel = QHBoxLayout()
            for index, item in enumerate(element):
                if index == 0:
                    item.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.horizontalLabel.addWidget(item)
            qw.setLayout(self.horizontalLabel)
            self.selectedLayout.addWidget(qw)
            self.selectedWidgets.append(qw)
        self.selectedLayout.addStretch()

        for index1, element in enumerate(self.selectImgRightLabels):
            for item in element:
                if index1 == len(self.selectImgRightLabels) - 1:
                    item.setFont(QFont("Arial", 40))
                else:
                    item.setFont(QFont("Arial", 34))

        self.show_right(self.selectedWidgets, True)
        self.setRightText(self.selectImgRightLabels)
        if cameraSection:
            return self.selectedLayout, self.selectedWidgets, self.selectImgRightLabels, self.overrideButton, self.wrongImg, self.recordingStatus, self.timeStatus
        return self.selectedLayout, self.selectedWidgets, self.selectImgRightLabels, self.overrideButton, self.wrongImg

    def show_right(self, section, shouldShow):
        for item in section:
            if shouldShow:
                item.show()
            else:
                item.hide()

    def setRightText(self, labels):
        if labels is None:
            return 
        for index1, element in enumerate(labels):
            if index1==1:
                pixmap = QPixmap('images/aiims.png')
                element[0]=QLabel()
                # element[0].setPixmap(self.convert_cv_qt())
                print(element)
                element[0].setPixmap(pixmap)
                continue
            for index2, element2 in enumerate(element):
                if index2 == 0:
                    element2.setText(self.placeholderText[index1])
                else:
                    element2.setText(self.scoreText[index1])
   
    def start_recording(self):
        global capture
        global img
        if self.model is None:
            self.showdialog(text1="Model loading...", text2="Model required to evaluate is loading. Please wait")
            return
        if capture is None:
            self.showdialog(text1="No capture device found", text2="Please select a valid capture device and try again later.")
            return
        if self.recordingStartBtn.text() == "Start Recording":
            file_name = QFileDialog.getSaveFileName(self, 'Save File')
            file_name = tuple(filter(lambda x:  x != '', file_name))
            if file_name is not None and len(file_name) > 1:
                file_name = file_name[0]
                self.th.start_recording(file_name)
                self.recordingStartBtn.setText("Stop Recording")
                self.tabs.tabBar().setEnabled(False)
                self.sourceBtn.setEnabled(False)
            else:
                self.showdialog(text1="Some error occured", text2="Some error occured while starting recording. Please try again.")
                return
        elif self.recordingStartBtn.text() == "Stop Recording":
            self.th.restart_recording()
            self.th.stop_recording()
            self.recordingStartBtn.setText("Start Recording")
            self.recordingPauseBtn.setText("Pause Recording")
            self.tabs.tabBar().setEnabled(True)
            self.sourceBtn.setEnabled(True)

    def pause_recoding(self):
        global capture
        if self.model is None:
            self.showdialog(text1="Model loading...", text2="Model required to evaluate is loading. Please wait")
            return
        if capture is None:
            self.showdialog(text1="No capture device found", text2="Please select a valid capture device and try again later.")
            return
        if self.th is not None and not self.th.is_recording():
            self.showdialog(text1="Not recording", text2="Please start a recording in order to pause it.")
            return
        if self.recordingPauseBtn.text() == "Pause Recording":
            self.th.pause_recording()
            self.recordingPauseBtn.setText("Restart Recording")
        elif self.recordingPauseBtn.text() == "Restart Recording":
            self.th.restart_recording()
            self.recordingPauseBtn.setText("Pause Recording")
    
    def capture_photo(self):
        global capture
        global img
        if self.tabs.currentIndex() == 0:
            self.photoImageSavedStatus.setText("")
            self.photoImageSavedStatus.hide()
        else:
            self.cameraImageSavedStatus.setText("")
            self.cameraImageSavedStatus.hide()
        if self.model is None:
            self.showdialog(text1="Model loading...", text2="Model required to evaluate is loading. Please wait")
            return
        if capture is None: 
            self.showdialog(text1="No input selected", text2="Please select an input device to capture an image")
            return
        _, image = capture.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        if not os.path.exists("data/temp"):
            os.mkdir("data/temp")
        cv2.imwrite("data/temp/temp.jpg", image)
        
        self.annotator = Annotator(image = "data/temp/temp.jpg", parent=self, callback=lambda x, y: self.recieved_camera_edited(x, y))
        self.annotator.show()

    def homographic_transformation(self, file, bbox, side=224):
        try:
            
            if not bbox or len(bbox) != 4:
                print("Error: Invalid bbox format - expecting 4 coordinates")
                return  None
                
            for i, coord in enumerate(bbox):
                if not isinstance(coord, (tuple, list)) or len(coord) != 2:
                    print(f"Error: Invalid coordinate format in bbox at index {i}")
                    return None
                try:        
                    float(coord[0])
                    float(coord[1])
                except (ValueError, TypeError):
                    print(f"Error: Non-numeric coordinate values at index {i}: {coord}")
                    return None
                    
            box = [list(bbox[0])]
            box.append(list(bbox[1]))
            box.append(list(bbox[2]))
            box.append(list(bbox[3]))
            
            try:
                rotated_box = np.array(box, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"Error: Invalid coordinate values: {e}")
                return None
            
            if len(np.unique(rotated_box, axis=0)) < 3:
                print("Error: Insufficient unique points for homography")
                return None
                
            target_box = np.array([(0, 0), (side, 0), (side, side), (0, side)], dtype=np.float32)
            try:
                H, mask = cv2.findHomography(rotated_box, target_box, cv2.RANSAC, 5.0)
            except Exception as e:
                print(f"Error calculating homography: {e}")
                return None
            
            if H is None:
                print("Error: Could not calculate homography matrix")
                return None
            
            try:
                image = cv2.imread(file)
                if image is None:
                    print(f"Error: Could not load image from {file}")
                    return None
            except Exception as e:
                print(f"Error loading image: {e}")
                return None                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                transformed_image = cv2.warpPerspective(image, H, (side, side))
                if len(transformed_image.shape) == 3 and transformed_image.shape[2] == 3:
                    pass
                else:
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error applying transformation: {e}")
                return None
            
            try:
                return Image.fromarray(transformed_image)
            except Exception as e:
                print(f"Error converting to PIL Image: {e}")
                return None
            
        except Exception as e:
            print(f"Unexpected error in homographic_transformation: {e}")
            return None

    def handle_crop_callback(self, img, bbox):
        global filename
        
        try:
            print(f"DEBUG: handle_crop_callback called with bbox: {bbox}")
            if img is None:
                print("Error: No image provided to callback")
                self.show_error_dialog("Error", "No image data received. Please try again.")
                return   
            if not bbox:
                print("Error: No bbox provided to callback")
                self.show_error_dialog("Error", "No crop area data received. Please try again.")
                return        
            if not filename:
                print("Error: No filename available")
                self.show_error_dialog("Error", "No image file loaded. Please load an image first.")
                return            
            result = self.homographic_transformation(filename, bbox)            
            if result is None:
                print("Error: homographic_transformation returned None")
                self.show_error_dialog("Crop Error", "Failed to process the cropped image. Please ensure you've selected a valid rectangular area and try again.")
                return                
            print("Crop processed successfully")
        except Exception as e:
            print(f"Unexpected error in handle_crop_callback: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_dialog("Error", "An unexpected error occurred while processing the crop. Please try again.")

    def show_error_dialog(self, title, message):
        """Centralized error dialog method for the main GUI"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(title)
            msg.setText(message)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setModal(True)            
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            result = msg.exec_()
            return result
        except Exception as e:
            print(f"Error showing dialog: {e}")

    def open_annotator(self):
        """Method to open the annotator with proper error handling"""
        try:
            global filename
            if not filename:
                self.show_error_dialog("Error", "Please load an image first.")
                return
            self.annotator = Annotator(filename, self, self.handle_crop_callback)
            self.annotator.show()
        except Exception as e:
            print(f"Error opening annotator: {e}")

    def recieved_camera_edited(self, image, bbox):
        global img
        global fileName
        global originalFileName
        global originalImg
        if self.tabs.currentIndex() == 0:
            x, y, w, h = bbox
            self.photobbox = bbox
            # image = image[y:y+h, x:x+w]
            print(fileName)
            image = self.homographic_transformation(fileName,bbox)
            image = np.asarray(image)[:,:,::-1]
            if not os.path.exists("data/temp"):
                os.mkdir("data/temp")
            #cv2.imwrite("data/temp/temp2.jpg", image)
            Image.fromarray(image).save("data/temp/temp2.jpg")
            fileName = "data/temp/temp2.jpg"
        else:
            originalImg = image
            self.camerabbox = bbox
            image = np.array(image)
            x, y, w, h = bbox
            print(fileName)
            image = self.homographic_transformation('data/temp/temp.jpg',bbox)
            image = np.asarray(image)[:,:,::-1]
            # image = image[y:y+h, x:x+w]
            img = image
        image_aspect_ratio = image.shape[1] / image.shape[0]
        height, width, channel = image.shape
        image = cv2.resize(image, (int(self.display_height * image_aspect_ratio), self.display_height), interpolation=cv2.INTER_CUBIC)
        height, width, channel = image.shape
        try:
            image = cv2.copyMakeBorder(image, int((self.display_height - height) / 2), int((self.display_height - height) / 2), int((self.display_width - width) / 2), int((self.display_width - width) / 2), cv2.BORDER_CONSTANT)
        except:
            self.showdialog(text1="some error occured", text2="Please try again")
            return
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        if self.tabs.currentIndex() == 0:
            self.selectedPhotoLbl2.setPixmap(QPixmap(qImg))
        else:
            self.cameraLbl2.setPixmap(QPixmap(qImg))
        self.annotator.hide()

    def open_file(self):
        global fileName
        global originalFileName
        if self.model is None:
            self.showdialog(text1="Model loading...", text2="Model required to evaluate is loading. Please wait")
            return
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if len(fname) != 0:
            try:
                image = cv2.imread(fname[0])
                fileName = fname[0]
                originalFileName = fname[0]
                image_aspect_ratio = image.shape[1] / image.shape[0]
                height, width, channel = image.shape
                image = cv2.resize(image, (int(self.display_height * image_aspect_ratio), self.display_height), interpolation=cv2.INTER_CUBIC)
                height, width, channel = image.shape
                image = cv2.copyMakeBorder(image, int((self.display_height - height) / 2), int((self.display_height - height) / 2), int((self.display_width - width) / 2), int((self.display_width - width) / 2), cv2.BORDER_CONSTANT)
                height, width, channel = image.shape
                bytesPerLine = 3 * width
                qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.selectedPhotoLbl.setPixmap(QPixmap(qImg))
                self.annotator = Annotator(image = fname[0], parent=self, callback=lambda x, y: self.recieved_camera_edited(x, y))
                self.annotator.show()
            except:
                self.showdialog(text1 = "No valid image found", text2 = "Please find a valid image to the program to get results.")
                print("Please select a valid file")               
        else:
            return

    def showdialog(self, text1, text2):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
    
    def showInputDialog(self):
        global originalImg
        global originalFileName
        global gradcamFileName
        
        if self.tabs.currentIndex() == 0 and originalFileName is None:
            self.showdialog(text1="Nothing evaluated",
                            text2="Please evaluate an image to generate report")
            return
        elif self.tabs.currentIndex() == 1 and originalImg is None:
            self.showdialog(text1="Nothing evaluated",
                            text2="Please evaluate an image to generate report")
            return
        
        msg = InputDialog(self)
        if msg.exec() and len(list(filter(lambda x: "__" in x, self.scoreText))) == 0:
            if self.tabs.currentIndex() == 0 and originalFileName is None:
                self.showdialog(text1="Some Error occured-1",
                                text2="Please try again later")
                return
            elif self.tabs.currentIndex() == 1 and originalImg is None:
                self.showdialog(text1="Some Error occured-2",
                                text2="Please try again later")
                return
        
            print("DEBUG: Saving original image...")
            if self.tabs.currentIndex() == 0:
                if isinstance(originalFileName, str):
                    print(f"DEBUG: Copying photo file: {originalFileName}")
                    shutil.copy2(originalFileName, "data/temp/report_screenshot.png")
                else:
                    print("DEBUG: Converting and saving photo array")
                    originalFileName = cv2.cvtColor(originalFileName, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("data/temp/report_screenshot.png", originalFileName)
            else:
                print("DEBUG: Converting and saving camera image")
                originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
                cv2.imwrite("data/temp/report_screenshot.png", originalImg)
            
            print("DEBUG: Starting GradCAM image processing...")
            gradcam_saved = False
            gradcam_path = None
            
            if self.tabs.currentIndex() == 1:  # Camera tab
                print("DEBUG: Processing camera tab GradCAM")
                print(f"DEBUG: gradcamFileName type: {type(gradcamFileName)}")
                print(f"DEBUG: gradcamFileName value: '{gradcamFileName}'")
                
                if gradcamFileName and gradcamFileName.strip() != "" and gradcamFileName != "None":
                    print(f"DEBUG: Checking if gradcam file exists: {gradcamFileName}")
                    if os.path.exists(gradcamFileName):
                        try:
                            gradcam_path = "data/temp/report_gradcam.png"
                            shutil.copy2(gradcamFileName, gradcam_path)
                            gradcam_saved = True
                            print(f"DEBUG: Successfully copied camera gradcam from {gradcamFileName} to {gradcam_path}")
                        except Exception as e:
                            print(f"ERROR: Failed to copy camera gradcam: {e}")
                    else:
                        print(f"ERROR: Camera gradcam file not found: {gradcamFileName}")
                else:
                    print("DEBUG: gradcamFileName is None, empty, or 'None'")
            else:  # Photo tab (index 0)
                print("DEBUG: Processing photo tab GradCAM")
                static_gradcam_file = "data/temp/gradcam_static.jpg"
                print(f"DEBUG: Checking if static gradcam file exists: {static_gradcam_file}")
                
                if os.path.exists(static_gradcam_file):
                    try:
                        gradcam_path = "data/temp/report_gradcam.png"
                        shutil.copy2(static_gradcam_file, gradcam_path)
                        gradcam_saved = True
                        print(f"DEBUG: Successfully copied static gradcam from {static_gradcam_file} to {gradcam_path}")
                    except Exception as e:
                        print(f"ERROR: Failed to copy static gradcam: {e}")
                else:
                    print(f"ERROR: Static gradcam file not found: {static_gradcam_file}")
            
            if gradcam_saved and gradcam_path:
                if os.path.exists(gradcam_path):
                    file_size = os.path.getsize(gradcam_path)
                    print(f"DEBUG: GradCAM file verified - Size: {file_size} bytes")
                else:
                    print("ERROR: GradCAM file was not actually saved!")
                    gradcam_saved = False
            
            outputs = msg.getInputs()
            
            if outputs == None:
                return
                
            file_name = outputs["video_path"]
            try:
                duration = self.get_time(file_name)
                time = str(datetime.timedelta(seconds=int(float(duration))))
                print("Time is: ", time)
            except:
                time = "No Recording"
        
            output_dict = {
                "name": outputs["name"],
                "email": outputs["email"],
                "program": outputs["program"],
                "iteration": outputs["iteration"],
                "item1": self.scoreText[0],
                "image": "data/temp/report_screenshot.png",
                "time": time     
            }

            if gradcam_saved:
                output_dict["gradImage"] = "data/temp/report_gradcam.png"
                print("DEBUG: Added gradImage to output dictionary")
            else:
                print("WARNING: No GradCAM image available for report")
                
            print("DEBUG: Final output dictionary:")
            print(output_dict)
            
            if "image" in output_dict and os.path.exists(output_dict["image"]):
                print(f"DEBUG: Main image file exists: {output_dict['image']}")
            else:
                print(f"ERROR: Main image file missing: {output_dict.get('image', 'N/A')}")
                
            if "gradImage" in output_dict and os.path.exists(output_dict["gradImage"]):
                print(f"DEBUG: GradCAM image file exists: {output_dict['gradImage']}")
            else:
                print(f"DEBUG: GradCAM image file missing or not included: {output_dict.get('gradImage', 'N/A')}")
            
            writer = PdfWrite(output_dict)
            writer.write()


    def cameraTabUI(self):
        """Create the Camera page UI."""
        self.cameraTab = QWidget()
        self.mainCameraLayout = QHBoxLayout()
        self.leftCameraLayout = QVBoxLayout()
        self.topButtonLayout = QHBoxLayout()

        self.captureCameraBtn = QPushButton("Capture Photo")
        self.captureCameraBtn.setFont(QFont("Arial", 16))
        self.captureCameraBtn.setStyleSheet("""
            QPushButton: {
                border-radius: 10px;
                border-color: black;
                border-width: 2px;
            }
        """)
        self.captureCameraBtn.clicked.connect(self.capture_photo)
        self.captureCameraBtn.setFixedHeight(40)

        self.recordingStartBtn = QPushButton("Start Recording")
        self.recordingStartBtn.setFont(QFont("Arial", 16))
        self.recordingStartBtn.setStyleSheet("""
            QPushButton: {
                border-radius: 10px;
                border-color: black;
                border-width: 2px;
            }
        """)
        self.recordingStartBtn.clicked.connect(self.start_recording)
        self.recordingStartBtn.setFixedHeight(40)

        self.recordingPauseBtn = QPushButton("Pause Recording")
        self.recordingPauseBtn.setFont(QFont("Arial", 16))
        self.recordingPauseBtn.setStyleSheet("""
            QPushButton: {
                border-radius: 10px;
                border-color: black;
                border-width: 2px;
            }
        """)
        self.recordingPauseBtn.clicked.connect(self.pause_recoding)
        self.recordingPauseBtn.setFixedHeight(40)

        self.cameraLbl = QLabel()
        self.cameraLbl.setPixmap(QPixmap("src/images/placeholder.jpg").scaled(self.display_width, self.display_height))
        self.cameraLayout = QHBoxLayout()
        self.cameraLayout.addStretch()
        self.cameraLayout.addWidget(self.cameraLbl)
        self.cameraLayout.addStretch()

        self.cameraLbl2 = QLabel()
        self.cameraLbl2.setPixmap(QPixmap("src/images/placeholder.jpg").scaled(self.display_width, self.display_height))
        self.cameraLayout2 = QHBoxLayout()
        self.cameraLayout2.addStretch()
        self.cameraLayout2.addWidget(self.cameraLbl2)
        self.cameraLayout2.addStretch()


        self.evaluatePhotoBtn = QPushButton("Evaluate")
        self.evaluatePhotoBtn.setFont(QFont("Arial", 16))
        self.evaluatePhotoBtn.clicked.connect(self.evaluate)
        self.evaluatePhotoBtn.setFixedHeight(40)

        self.topButtonLayout.addWidget(self.captureCameraBtn)
        self.topButtonLayout.addWidget(self.recordingStartBtn)
        self.topButtonLayout.addWidget(self.recordingPauseBtn)
        self.leftCameraLayout.addLayout(self.topButtonLayout)
        self.leftCameraLayout.addLayout(self.cameraLayout)
        self.leftCameraLayout.addLayout(self.cameraLayout2)
        self.leftCameraLayout.addWidget(self.evaluatePhotoBtn)

        self.rightCameraLayout, self.rightCameraWidgets, self.rightCameraLabels, self.cameraOverrideButton, self.cameraWrongLbl, self.cameraStatus, self.timeStatus = self.setupSelectLabels(num_labels=6, cameraSection=True)
        self.cameraOverrideButton.clicked.connect(self.overrideEval)
        self.rightCameraLayout.setContentsMargins(20, 100, 20, 100)

        self.resultCameraBox = QHBoxLayout()

        self.GradcamCamerabutton = QPushButton("Grad Cam")
        self.GradcamCamerabutton.setFont(QFont("Arial", 16))
        self.GradcamCamerabutton.clicked.connect(self.gradcam)
        self.GradcamCamerabutton.setFixedHeight(40)
        
        self.satisfiedCameraBtn = QPushButton("Satisfied")
        self.satisfiedCameraBtn.setFont(QFont("Arial", 16))
        self.satisfiedCameraBtn.clicked.connect(self.satisfied)
        self.satisfiedCameraBtn.setFixedHeight(40)
        self.notSatisfiedCameraBtn = QPushButton("Not Satisfied")
        self.notSatisfiedCameraBtn.setFont(QFont("Arial", 16))
        self.notSatisfiedCameraBtn.clicked.connect(self.unsatisfied)
        self.notSatisfiedCameraBtn.setFixedHeight(40)
        self.cameraImageSavedStatus = QLabel()
        self.cameraImageSavedStatus.setAlignment(Qt.AlignCenter)
        self.cameraImageSavedStatus.hide()

        self.cameraGenerateBtn = QPushButton("Generate Report")
        self.cameraFeedbackBtn = QPushButton("Submit Feedback")
        self.cameraGenerateBtn.setFont(QFont("Arial", 16))
        self.cameraGenerateBtn.clicked.connect(self.showInputDialog)
        self.cameraGenerateBtn.setFixedHeight(40)
        self.cameraFeedbackBtn.setFont(QFont("Arial", 16))
        self.cameraFeedbackBtn.clicked.connect(self.feedback)
        self.cameraFeedbackBtn.setFixedHeight(40)

        self.scoreCameraField = QLineEdit()
        self.scoreCameraField.setFixedWidth(100)
        self.scoreCameraField.setFixedHeight(40)

        self.resultCameraBox.addWidget(self.satisfiedCameraBtn)
        self.resultCameraBox.addWidget(self.notSatisfiedCameraBtn)
        self.resultCameraBox.addWidget(self.cameraImageSavedStatus)
        self.resultCameraBox.addWidget(self.scoreCameraField)
        self.resultCameraBox.addWidget(self.GradcamCamerabutton)

        self.cameraExtraButtonsLayout = QHBoxLayout()
        self.cameraExtraButtonsLayout.addWidget(self.cameraGenerateBtn)
        self.cameraExtraButtonsLayout.addWidget(self.cameraFeedbackBtn)

        self.rightCameraLayout.addLayout(self.cameraExtraButtonsLayout)
        self.rightCameraLayout.addLayout(self.resultCameraBox)
        
        self.mainCameraLayout.addLayout(self.leftCameraLayout, 50)
        self.mainCameraLayout.addLayout(self.rightCameraLayout, 50)
        self.mainCameraLayout.setSpacing(40)
        self.cameraTab.setLayout(self.mainCameraLayout)
        return self.cameraTab

    def gradcam(self):
        global fileName
        global gradcamFileName
        global preEvaluateGradcam
        
        # Use current tab instead of button states
        if self.tabs.currentIndex() == 1:  # Camera tab
     
            if preEvaluateGradcam == 0:
                self.showdialog(text1="No GradCam available", text2 = "Please evaluate a camera image first")
                return 

            if not gradcamFileName or gradcamFileName.strip() == "" or gradcamFileName == "None":
                print("ERROR: gradcamFileName is invalid!")
                self.showdialog(text1="No GradCAM available", text2="Please evaluate a camera image first")
                return
                
            print(f"DEBUG: About to read camera gradcam file: '{gradcamFileName}'")
            print(f"DEBUG: fileName = '{fileName}'")
            print(f"DEBUG: Current tab index = {self.tabs.currentIndex()}")
            
            if not os.path.exists(gradcamFileName):
                print(f"ERROR: Camera gradcam file does not exist: {gradcamFileName}")
                self.showdialog(text1="GradCAM file not found", text2="Please evaluate the camera image again")
                return
                
            image = cv2.imread(gradcamFileName)
            
            if image is None:
                print(f"ERROR: Could not load camera gradcam image from: {gradcamFileName}")
                self.showdialog(text1="Could not load GradCAM image", text2="Please try again")
                return
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            image_aspect_ratio = image.shape[1] / image.shape[0]
            height, width, channel = image.shape
            image = cv2.resize(image, (int(self.display_height * image_aspect_ratio), self.display_height), interpolation=cv2.INTER_CUBIC)
            height, width, channel = image.shape
            try:
                image = cv2.copyMakeBorder(image, int((self.display_height - height) / 2), int((self.display_height - height) / 2), int((self.display_width - width) / 2), int((self.display_width - width) / 2), cv2.BORDER_CONSTANT)
            except:
                self.showdialog(text1="some error occured", text2="Please try again")
                return
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.cameraLbl2.setPixmap(QPixmap(qImg))
            
        else:  

            static_gradcam_file = "data/temp/gradcam_static.jpg"

            if preEvaluateGradcam == 0:
                self.showdialog(text1="No GradCam available", text2 = "Please evaluate a image first")
                return
            if not os.path.exists(static_gradcam_file):
                print(f"ERROR: Static gradcam file does not exist: {static_gradcam_file}")
                self.showdialog(text1="No GradCAM available", text2="Please evaluate a photo first")
                return
            print(f"DEBUG: About to read static gradcam file: '{static_gradcam_file}'")
            image = cv2.imread(static_gradcam_file)
            if image is None:
                print(f"ERROR: Could not load static gradcam image from: {static_gradcam_file}")
                self.showdialog(text1="Could not load GradCAM image", text2="Please try again")
                return    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_aspect_ratio = image.shape[1] / image.shape[0]
            height, width, channel = image.shape
            image = cv2.resize(image, (int(self.display_height * image_aspect_ratio), self.display_height), interpolation=cv2.INTER_CUBIC)
            height, width, channel = image.shape
            try:
                image = cv2.copyMakeBorder(image, int((self.display_height - height) / 2), int((self.display_height - height) / 2), int((self.display_width - width) / 2), int((self.display_width - width) / 2), cv2.BORDER_CONSTANT)
            except:
                self.showdialog(text1="some error occured", text2="Please try again")
                return
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.selectedPhotoLbl2.setPixmap(QPixmap(qImg))


    def get_frames(self, file_path):
        probe = ffmpeg.probe(file_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        return int(video_stream["nb_frames"])

    def get_time(self, file_path):
        probe = ffmpeg.probe(file_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        return video_stream["duration"]

    def convertQImageToMat(self, incomingImage):
        incomingImage = incomingImage.convertToFormat(QImage.Format_RGBX8888)
        ptr = incomingImage.constBits()
        ptr.setsize(incomingImage.byteCount())
        cv_im_in = np.array(ptr, copy=True).reshape(incomingImage.height(), incomingImage.width(), 4)
        cv_im_in = cv2.cvtColor(cv_im_in, cv2.COLOR_BGRA2RGB)
        return cv_im_in

    def closeEvent(self, event):
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.cameraLbl.setPixmap(qt_img)
    
    @pyqtSlot(str)
    def update_timestamp(self, time):
        self.cameraStatus.show()
        if self.th._recording and not self.th._pause_recording:
            ## Set it to recording 
            self.cameraStatus.setText("Recording")
            self.timeStatus.show()
            self.timeStatus.setText(time)
            self.cameraStatus.setStyleSheet("""
                QLabel#red{
                    background-color: green;
                    border-radius: 10px;        
                    border-width: 2px;
                    border-color: #000000;
                    border-style: solid;
                }
            """)
        elif self.th._recording and self.th._pause_recording:
            ## Set it to paused recording
            self.cameraStatus.setText("Paused Recording")
            self.timeStatus.show()
            self.timeStatus.setText(time)
            self.cameraStatus.setStyleSheet("""
                QLabel#red{
                    background-color: orange;
                    border-radius: 10px;        
                    border-width: 2px;
                    border-color: #000000;
                    border-style: solid;
                }
            """)
        else:
            ## Set it to no recording
            self.cameraStatus.setText("Not Recording")
            self.timeStatus.hide()
            self.timeStatus.setText("NA")
            self.cameraStatus.setStyleSheet("""
                QLabel#red{
                    background-color: red;
                    border-radius: 10px;        
                    border-width: 2px;
                    border-color: #000000;
                    border-style: solid;
                }
            """)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height)
        return QPixmap.fromImage(p)
    
    def get_aspect_ratio(self):
        global capture
        _, frame = capture.read()
        return frame.shape[1] / frame.shape[0]

if __name__ == "__main__":
    os_used = sys.platform
    process = psutil.Process(os.getpid())  # Set highest priority for the python script for the CPU
    if os_used == "win32":  # Windows (either 32-bit or 64-bit)
        process.nice(psutil.REALTIME_PRIORITY_CLASS)
    else:  # MAC OS X and Linux
        os.system("sudo renice -n -20 -p " + str(os.getpid()))

    app = QApplication(sys.argv)
    window = Application()
    # window.setWindowFlags(Qt.WindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint))
    window.showFullScreen()
    window.show()
    sys.exit(app.exec())
