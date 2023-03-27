import sys
import pose_media as pm
import mediapipe as mp
import numpy as np
import torch

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage,QPixmap,QBitmap
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import time


class Ui_Dialog(object):
    threshold = 0
    actions = np.array(["walking", "boxing", "handwaving"])
    pTime = 0
    cTime = 0

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(838, 616)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(42, 32, 781, 571))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.video_click = QtWidgets.QPushButton(self.widget, clicked=lambda: self.video_cap())
        self.video_click.setObjectName("video_click")
        self.gridLayout.addWidget(self.video_click, 2, 1, 1, 1)
        self.label_load_action = QtWidgets.QLabel(self.widget)
        self.label_load_action.setObjectName("label_load_action")
        self.gridLayout.addWidget(self.label_load_action, 0, 2, 1, 1)
        self.label_FPS = QtWidgets.QLabel(self.widget)
        self.label_FPS.setObjectName("label_FPS")
        self.gridLayout.addWidget(self.label_FPS, 1, 2, 1, 1)
        self.webcam_click = QtWidgets.QPushButton(self.widget, clicked=lambda: self.web_cam())
        self.webcam_click.setObjectName("webcam_click")
        self.gridLayout.addWidget(self.webcam_click, 2, 0, 1, 1)
        self.label_load_video = QtWidgets.QLabel(self.widget)
        self.label_load_video.setObjectName("label_load_video")
        self.gridLayout.addWidget(self.label_load_video, 0, 0, 2, 2)

        self.pose = pm.mediapipe_pose()
        self.pt = self.pose.mp_holistic.Holistic()
        self.new_model = torch.load('model.pth')
        self.new_model.eval()

        self.sequence = []
        self.sentence = []

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def web_cam(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                frame, results = self.pose.mediapipe_detection(frame, self.pt)
            except:
                pass
            self.pose.draw_styled_landmarks(frame, results)
            keypoints = self.pose.extract_keypoints(results)
            # frame.shape摄像头捕获的图像480x640x3
            # keypoints关键点132维度
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            # print(np.expand_dims(self.sequence, axis=0).shape)
            if len(self.sequence) == 30:
                inputs = torch.from_numpy(np.expand_dims(self.sequence, axis=0)[0]).unsqueeze(0).to(torch.float32)
                inputs = inputs.repeat(2, 1, 1)
                res = self.new_model(inputs)[0]
                res = res.detach().numpy()
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence) > 0:
                        if self.actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(self.actions[np.argmax(res)])
                    else:
                        self.sentence.append(self.actions[np.argmax(res)])
                if len(self.sentence) > 1:
                    self.sentence = self.sentence[-1:]
            self.label_load_action.setText(' '.join(self.sentence))
            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            self.label_FPS.setText("FPS: " + str(int(fps)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            convertToQtFormat = QtGui.QPixmap.fromImage(image)
            pixmap = QPixmap(convertToQtFormat)
            resizeImage = pixmap.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            QApplication.processEvents()
            self.label_load_video.setPixmap(resizeImage)

    def video_cap(self):
        path = QtWidgets.QFileDialog.getOpenFileName()[0]
        if path:
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                frame, results = self.pose.mediapipe_detection(frame, self.pt)
                self.pose.draw_styled_landmarks(frame, results)
                keypoints = self.pose.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    inputs = torch.from_numpy(np.expand_dims(self.sequence, axis=0)[0]).unsqueeze(0).to(torch.float32)
                    inputs = inputs.repeat(2, 1, 1)
                    res = self.new_model(inputs)[0]
                    res = res.detach().numpy()
                    if res[np.argmax(res)] > self.threshold:
                        if len(self.sentence) > 0:
                            if self.actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(self.actions[np.argmax(res)])
                        else:
                            self.sentence.append(self.actions[np.argmax(res)])
                    if len(self.sentence) > 1:
                        self.sentence = self.sentence[-1:]
                self.label_load_action.setText(' '.join(self.sentence))
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime
                self.label_FPS.setText("FPS: " + str(int(fps)))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                convertToQtFormat = QtGui.QPixmap.fromImage(image)
                pixmap = QPixmap(convertToQtFormat)
                resizeImage = pixmap.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                QApplication.processEvents()
                self.label_load_video.setPixmap(resizeImage)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.video_click.setText(_translate("Dialog", "Video"))
        self.label_load_action.setText(_translate("Dialog", "TextLabel"))
        self.label_FPS.setText(_translate("Dialog", "TextLabel"))
        self.webcam_click.setText(_translate("Dialog", "Webcam"))
        self.label_load_video.setText(_translate("Dialog", "TextLabel"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())