from PyQt5 import QtCore, QtGui
from pyqtgraph import PlotWidget
from FlirWindow import Ui_MainWindow
from fitgauss import fitgauss2d_section, gauss1d
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt




class Ui_CustomWindow(Ui_MainWindow):
    def custom_init(self,mainwindow):
        self.mainwindow = mainwindow
        self.section_xctr = 0
        self.section_yctr = 0
        self.section_xdata = []
        self.section_ydata = []
        self.section_xcoord = []
        self.section_ycoord = []
        self.save_dir = os.getcwd()
        self.unit = 0

        #logo
        mainwindow.setWindowIcon(QtGui.QIcon('logo.png'))

        # logging
        self.cam_controller.log = self.plainTextEditLog

        #init
        self.cam_controller.initialize()

        # init start and stop continue button
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_movie)
        self.pushButtonContinue.clicked.connect(self.start_continue)

        # init exposure time line edit
        self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
        self.lineEditExposureTime.returnPressed.connect(self.set_exptime)

        # init section plots
        self.plotx = PlotWidget(self.centralwidget)
        self.plotx.setObjectName("plotx")
        self.gridLayoutImage.addWidget(self.plotx, 1, 0, 1, 1)
        self.sectionx_line = self.plotx.plot(self.section_xcoord,self.section_xdata)

        self.ploty = PlotWidget(self.centralwidget)
        self.ploty.setObjectName("ploty")
        self.gridLayoutImage.addWidget(self.ploty,  0, 1, 1, 1)
        self.sectiony_line = self.ploty.plot(self.section_ydata,self.section_ycoord)

        # set section to the center
        self.pushButtonSectionCenter.clicked.connect(self.section_center)

        # set section center by line edit
        self.lineEditSectionX.returnPressed.connect(self.section_center_line_edit)
        self.lineEditSectionY.returnPressed.connect(self.section_center_line_edit)

        # image label mouse press event
        self.labelImage.mousePressEvent = self.label_mousepress()

        # auto exposure check box
        self.checkbox_auto_exposure()
        self.checkBoxAutoExposure.stateChanged.connect(self.checkbox_auto_exposure)

        # background
        self.pushButtonSetBg.clicked.connect(self.set_background)
        self.pushButtonClearBg.clicked.connect(self.clear_background)

        # average frames
        self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
        self.lineEditAverageFrames.returnPressed.connect(self.set_average_frames)

        # save file
        self.actionsave_image_2.setStatusTip('Save File')
        self.actionsave_image_2.triggered.connect(self.file_save)

        # unit
        self.unit_change()
        self.radioButtonUnitPixel.toggled.connect(self.unit_change)

        # temperature monitor
        self.temperature_timer = QtCore.QTimer()
        self.temperature_timer.timeout.connect(self.update_temperature)
        self.temperature_timer.start(1000)

    def start_continue(self):
        self.cam_controller.start_continue()
        self.pushButtonContinue.setText("Stop Continue")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.stop_continue)
        # self.t0 = time.time()
        self.update_timer.start(500)
        # print('click')

    def stop_continue(self):
        self.update_timer.stop()
        # print("--- %.8f seconds ---" % (time.time() - self.t0))
        # print("no frames : %d"%(self.cam_controller.framecount))
        self.cam_controller.stop_continue()
        self.pushButtonContinue.setText("Start Continue")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.start_continue)

    def update_movie(self):
        self.cam_controller.acquire_continue()
        self.update_plot()
        p, ier = fitgauss2d_section(np.arange(0, self.cam_controller.frame.shape[1]),
                                    np.arange(0, self.cam_controller.frame.shape[0]), self.cam_controller.frame)
        self.lineEditxCenter.setText('%.4f' % (p[0]*self.unit))
        self.lineEdityCenter.setText('%.4f' % (p[1]*self.unit))
        self.lineEditxWaist.setText('%.4f' % (p[2]*2*self.unit))
        self.lineEdityWaist.setText('%.4f' % (p[3]*2*self.unit))
        self.lineEditHeight.setText('%.4f' % (p[4]))
        self.labelImage.setPixmap(QtGui.QPixmap(self.toQImage()))
        # plt.plot(self.cam_controller.frame[round(p[1]),::])
        # plt.plot(gauss1d(p[0],p[2],p[4],np.arange(0, self.cam_controller.frame.shape[1]))+p[5])
        # plt.show()
        if self.checkBoxAutoExposure.isChecked():
            self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))

    def update_plot(self):
        self.section_xcoord = np.arange(0, self.cam_controller.frame.shape[1])
        self.section_xdata = self.cam_controller.frame[self.section_yctr,::]
        self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
        self.section_ycoord = np.arange(0, self.cam_controller.frame.shape[0])
        self.section_ydata = self.cam_controller.frame[::,self.section_xctr]
        self.sectiony_line.setData(self.section_ydata,self.section_ycoord)

    def toQImage(self, copy=False):
        '''
        Transfer the format of the frame from numpy.ndarray to QImage
        :param self:
        :param copy:
        :return:
        '''
        qim = QtGui.QImage(self.cam_controller.frame.data, self.cam_controller.frame.shape[1],
                           self.cam_controller.frame.shape[0], self.cam_controller.frame.strides[0],
                           QtGui.QImage.Format_Indexed8).rgbSwapped()
        # qim.setColorTable(gray_color_table)
        return qim.copy() if copy else qim

    def set_exptime(self):
        self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
        self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))

    def label_mousepress(self):
        def mousepress(eventQMouseEvent):
            label_width = self.labelImage.size().width()
            label_height = self.labelImage.size().height()
            mouse_x = eventQMouseEvent.pos().x()
            mouse_y = eventQMouseEvent.pos().y()
            self.section_xctr = round(self.cam_controller.framewidth*mouse_x/label_width)
            self.section_yctr = round(self.cam_controller.frameheight*mouse_y/label_height)
            self.lineEditSectionX.setText(str(self.section_xctr))
            self.lineEditSectionY.setText(str(self.section_yctr))
            self.update_plot()
        return mousepress

    def section_center(self):
        self.section_xctr = round(float(self.lineEditxCenter.text())/self.unit)
        self.section_yctr = round(float(self.lineEdityCenter.text())/self.unit)
        self.lineEditSectionX.setText(str(self.section_xctr))
        self.lineEditSectionY.setText(str(self.section_yctr))

    def section_center_line_edit(self):
        try:
            self.section_xctr = round(float(self.lineEditSectionX.text())/self.unit)
            self.section_yctr = round(float(self.lineEditSectionY.text())/self.unit)
        except ValueError as ex:
            self.lineEditSectionX(str(self.section_xctr))
            self.lineEditSectionY(str(self.section_yctr))
            self.plainTextEditLog.insertPlainText('ValueError: %s' %(ex))

    def checkbox_auto_exposure(self):
        if self.checkBoxAutoExposure.isChecked():
            self.cam_controller.reset_exposure()
        else:
            self.set_exptime()

    def set_background(self):
        self.cam_controller.set_background()

    def clear_background(self):
        self.cam_controller.clear_background()

    def set_average_frames(self):
        if not self.cam_controller.set_average_frames(self.lineEditAverageFrames.text()):
            self.lineEditAverageFrames.setText(str(self.cam_controller.average_frames))

    def file_save(self):
        frametosave = self.cam_controller.frame
        filename = "cx"+self.lineEditxCenter.text()+"cy"+self.lineEdityCenter.text() \
            + "wx" + self.lineEditxWaist.text() + "wy" +self.lineEdityWaist.text() \
            + "h" + self.lineEditHeight.text()
        if self.radioButtonUnitPixel.isChecked():
            filename += "pixel"
        else:
            filename += "um"
        filename  += ".jpg"
        name = QtGui.QFileDialog.getSaveFileName(self.mainwindow, 'Save File',os.path.join(self.save_dir,filename),"Images (*.png *.jpg)")[0]
        if not name is '':
            self.save_dir = os.path.split(name)[0]
            cv2.imwrite(name,frametosave)
            self.plainTextEditLog.insertPlainText('Picture saved to %s\n'%(name))

    def unit_change(self):
        if self.radioButtonUnitPixel.isChecked():
            self.unit = 1
        else:
            self.unit = self.cam_controller.pixel_size

    def update_temperature(self):
        self.cam_controller.get_temperature()
        if self.cam_controller.device_temperature > self.cam_controller.device_temp_lim:
            self.statusbar.showMessage("Warning: device temperature too high! T = %.1f > %.1f"%(self.cam_controller.device_temperature,self.cam_controller.device_temp_lim))