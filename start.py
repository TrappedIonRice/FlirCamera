from FlirCamController import FlirCamController
from FlirWindowModify import Ui_CustomWindow
from FlirCamFake import FakeCamContoller
import qdarkstyle
import sys
from PyQt5 import QtWidgets
import time

flir = FlirCamController()
# flir = FakeCamContoller() #use to test the program without a camera

app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
MainWindow = QtWidgets.QMainWindow()
ui = Ui_CustomWindow()
ui.cam_controller = flir
ui.setupUi(MainWindow)
ui.custom_init(MainWindow)
MainWindow.show()
app.exec_()
flir.stop_continue()
flir.close()
sys.exit()


