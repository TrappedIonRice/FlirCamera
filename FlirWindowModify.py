import datetime
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtWidgets import QScrollBar, QScrollArea, QWidget, QApplication
from pyqtgraph import PlotWidget
from pyqtgraph import mkPen
from FlirWindow import Ui_MainWindow
from fitgauss import fitgauss2d_section, gauss1d, fitgauss2d_multiple
import numpy as np
import os
import cv2
import math

import matplotlib.pyplot as plt




class Ui_CustomWindow(Ui_MainWindow):
    def custom_init(self,mainwindow):
        self.mainwindow = mainwindow
        self.stime = time.time()
        self.nframe = 0
        self.lasttime = time.time()
        self.section_xctr = 0
        self.section_yctr = 0
        self.section_xctr0 = 0
        self.section_yctr0 = 0
        self.section_xdata = []
        self.section_ydata = []
        self.section_xcoord = []
        self.section_ycoord = []
        self.save_dir = os.getcwd()
        self.unit = 0
        self.iter = -1

        self.loclist = []
        self.zoom_x = 0
        self.zoom_y = 0
        self.current_scale = 0
        self.target_scale = 0
        self.zoom_scale_x = 0.0
        self.zoom_scale_y = 0.0

        # multiple fits line edit
        self.num_fits = 1
        self.lineEditMultiFits.setText('1')
        self.line_edit_multi_fits()
        # self.lineEditMultiFits.stateChanged.connect(self.line_edit_multi_fits)

        self.slices = 40
        # self.lineEditSlices.setText('40')
        # self.line_edit_slices()

        # Data for the fitted gaussian
        self.section_xdata_fit = []
        self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
        self.section_ydata_fit = []
        self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]


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

        self.colors = [(np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(100, 256))
                       for _ in range(self.num_fits)]

        self.sectionx_fit = self.plotx.plot(self.section_xcoord, self.section_xdata_fit,
                                            pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
        self.sectionx_fit_mult = [self.plotx.plot(self.section_xcoord, self.section_xdata_fit_mult[i],
                                                  pen=mkPen(color=self.colors[i], style=QtCore.Qt.DashLine))
                                  for i in range(len(self.section_xdata_fit_mult))]

        self.sectionx_line = self.plotx.plot(self.section_xcoord,self.section_xdata)

        self.ploty = PlotWidget(self.centralwidget)

        self.ploty.setObjectName("ploty")
        self.gridLayoutImage.addWidget(self.ploty,  0, 1, 1, 1)
        self.ploty.getPlotItem().getViewBox().invertY(True)

        self.sectiony_fit = self.ploty.plot(self.section_ydata_fit, self.section_ycoord,
                                            pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
        self.sectiony_fit_mult = [self.ploty.plot(self.section_ydata_fit_mult[i], self.section_ycoord,
                                                  pen=mkPen(color=self.colors[i], style=QtCore.Qt.DashLine))
                                  for i in range(len(self.section_ydata_fit_mult))]

        self.sectiony_line = self.ploty.plot(self.section_ydata,self.section_ycoord)

        # Display fit parameters for multiple fits in colors that match the plots
        self.param_displays = [[] for _ in range(5)]
        self.param_labels = ('xCenter', 'yCenter', 'xWaist', 'yWaist', 'Height')
        self.param_layout = QtWidgets.QGridLayout()
        self.param_layout.setObjectName('gridLayoutParameters')
        self.param_label_layout = QtWidgets.QGridLayout()
        self.param_label_layout.setObjectName('gridLayoutParameterLabels')
        # if True:
        #     self.param_displays = [[QtWidgets.QLabel(self.centralwidget) for _ in range(self.num_fits)] for _ in range(5)]
        #     for i in range(len(self.param_displays)):
        #         for j in range(len(self.param_displays[i])):
        #             self.param_displays[i][j].setFont(self.param_font)
        #             self.param_displays[i][j].setObjectName("display" + self.param_labels[i] + str(j + 1))
        #             self.param_displays[i][j].setStyleSheet('color: rgb(' + str(self.colors[j][0]) + ', '
        #                                                     + str(self.colors[j][1]) + ', ' + str(self.colors[j][2]) + ');')
        #             self.gridLayoutFitResult.addWidget(self.param_displays[i][j], i, j + 1, 1, 1)

        # set section to the center
        self.pushButtonSectionCenter.clicked.connect(self.section_center)

        # set section center by line edit
        self.lineEditSectionX.returnPressed.connect(self.section_center_line_edit)
        self.lineEditSectionY.returnPressed.connect(self.section_center_line_edit)

        # image label mouse press event
        self.labelImage.mousePressEvent = self.label_mousepress()

        # Image label mouse scroll event?
        self.labelImage.wheelEvent = self.label_mousewheel()

        # Zoom rectangle
        self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
        self.zoom_needed = False

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

        # center logging
        self.start_date = str(datetime.datetime.now()).replace(':', '-')
        self.start_time = time.time()
        if not os.path.exists('center_log'):
            os.makedirs('center_log')
        with open('center_log/' + self.start_date + '.txt', 'w') as f:
            f.write('time, x0, y0, x1, y1...\n')

    def start_continue(self):
        self.cam_controller.start_continue()
        self.pushButtonContinue.setText("Stop Continue")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.stop_continue)
        # self.t0 = time.time()
        self.update_timer.start(10) # It delays the next update of the window by 10ms.
        # Originally 500ms, that's the main reason why it has low frame rate.
        # print('click')
        self.iter = 0
        self.stime = time.time()

    def stop_continue(self):
        self.update_timer.stop()
        # print("--- %.8f seconds ---" % (time.time() - self.t0))
        # print("no frames : %d"%(self.cam_controller.framecount))
        self.cam_controller.stop_continue()
        self.pushButtonContinue.setText("Start Continue")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.start_continue)

    def update_movie(self):
        # stime=time.time() # debugging

        self.cam_controller.acquire_continue()
        # print('1:' + str((time.time() - stime) * 1000))
        if self.checkBoxFit.isChecked():
            self.line_edit_multi_fits()
            # self.line_edit_slices()
            # print('2:' + str((time.time() - stime) * 1000))
            if self.num_fits <= 1 or self.iter <= 0:
                print
                self.p, self.ier = fitgauss2d_section(np.arange(0, self.cam_controller.frame.shape[1]),
                                            np.arange(0, self.cam_controller.frame.shape[0]), self.cam_controller.frame)
                # print('3:' + str((time.time() - stime) * 1000))
                if self.num_fits <= 1:
                    self.update_plot()
                    # Display measurements in pixels instead of microns (if we want to convert back to microns, multiply by self.unit)
                    if self.lineEditxCenter.parent() is not None:
                        self.lineEditxCenter.setText('%.1f' % (self.p[0]))
                        self.lineEdityCenter.setText('%.1f' % (self.p[1]))
                        self.lineEditxWaist.setText('%.1f' % (self.p[2] * 2*self.unit))
                        self.lineEdityWaist.setText('%.1f' % (self.p[3] * 2*self.unit))
                        self.lineEditHeight.setText('%.1f' % (self.p[4]))
                    else:
                        self.update_param_displays()
                # print('4:' + str((time.time() - stime) * 1000))
                if self.iter <= 0:
                    if self.p[2] < self.cam_controller.framewidth / self.slices and \
                            self.p[3] < self.cam_controller.frameheight / self.slices:
                        self.zoom_needed = True
                        self.plainTextEditLog.insertPlainText(
                            'Please zoom in to the features for multiple logging to work\n')
                    else:
                        self.zoom_needed = False
            else:
                if not self.zoom_needed:
                    self.p = fitgauss2d_multiple(self.cam_controller.frame, np.arange(self.cam_controller.frame.shape[0]),
                                                 np.arange(self.cam_controller.frame.shape[1]), num_fits=self.num_fits,
                                                 slices=self.slices)
                else:
                    self.p = fitgauss2d_multiple(self.cam_controller.frame[self.rect.top():self.rect.bottom() + 1,
                                                 self.rect.left():self.rect.right() + 1], np.arange(self.rect.height()),
                                                 np.arange(self.rect.width()), num_fits=self.num_fits,
                                                 slices=self.slices)
                    for i in range(len(self.p)):
                        self.p[i][0] += self.rect.left()
                        self.p[i][1] += self.rect.top()

                self.update_plot()

                if self.lineEditxCenter.parent() is not None:
                    self.gridLayoutFitResult.addLayout(self.param_label_layout, 0, 0, 1, 1)
                    self.gridLayoutFitResult.addLayout(self.param_layout, 0, 1, 1, 1)
                    self.lineEditxCenter.setParent(None)
                    self.lineEdityCenter.setParent(None)
                    self.lineEditxWaist.setParent(None)
                    self.lineEdityWaist.setParent(None)
                    self.lineEditHeight.setParent(None)

                    self.labelxCenter.setParent(None)
                    self.param_label_layout.addWidget(self.labelxCenter, 0, 0, 1, 1)
                    self.labelyCenter.setParent(None)
                    self.param_label_layout.addWidget(self.labelyCenter, 1, 0, 1, 1)
                    self.labelxWaist.setParent(None)
                    self.param_label_layout.addWidget(self.labelxWaist, 2, 0, 1, 1)
                    self.labelyWaist.setParent(None)
                    self.param_label_layout.addWidget(self.labelyWaist, 3, 0, 1, 1)
                    self.labelHeight.setParent(None)
                    self.param_label_layout.addWidget(self.labelHeight, 4, 0, 1, 1)

                self.update_param_displays()

        else:
            self.update_plot_withoutfit()
            self.lineEditxCenter.setText('N/A')
            self.lineEdityCenter.setText('N/A')
            self.lineEditxWaist.setText('N/A')
            self.lineEdityWaist.setText('N/A')
            self.lineEditHeight.setText('N/A')

        self.iter += 1

        self.pixmap = QtGui.QPixmap(self.toQImage())
        self.zoom()


        #self.zoom_scale_x= float(self.pixmap.width())/float(self.cam_controller.framewidth)  #ratio of the displayed image to the original image
        #self.zoom_scale_y=float(self.pixmap.height())/float(self.cam_controller.frameheight)  #inequality in zoom_scale_x and zoom_scale_y may indicate that the image is zoomed to very few pixels
        #print(self.zoom_scale_x)
        #print(self.zoom_scale_y)
        if self.section_xctr != 0 and self.section_yctr != 0 and self.checkBoxCrosshair.isChecked():
            self.draw_crosshair(self.mouse_x,self.mouse_y)
        self.labelImage.setPixmap(self.pixmap)

        # plt.plot(self.cam_controller.frame[round(p[1]),::])
        # plt.plot(gauss1d(p[0],p[2],p[4],np.arange(0, self.cam_controller.frame.shape[1]))+p[5])
        # plt.show()

        if self.checkBoxAutoExposure.isChecked():
            self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
        # self.nframe += 1
        # print('5:' + str((time.time() - stime) * 1000))
        # print(time.time()-self.lasttime)
        # self.lasttime=time.time()
        # print((time.time() - self.stime) * 1000 / self.nframe)

    def update_param_displays(self, update_vals=True):
        while self.num_fits > len(self.param_displays[0]):
            for i in range(5):
                self.param_displays[i].append(QtWidgets.QLabel(self.centralwidget))
                self.param_displays[i][-1].setFont(self.param_font)
                self.param_displays[i][-1].setObjectName("display" + self.param_labels[i] +
                                                         str(len(self.param_displays[i])))
                self.param_displays[i][-1].setStyleSheet('color: rgb(' +
                                                         str(self.colors[len(self.param_displays[i]) - 1][0])
                                                         + ', ' +
                                                         str(self.colors[len(self.param_displays[i]) - 1][1])
                                                         + ', ' +
                                                         str(self.colors[len(self.param_displays[i]) - 1][2])
                                                         + '); background-color: black')
                self.param_layout.addWidget(self.param_displays[i][-1], i, len(self.param_displays[i]),
                                            1, 1)
        while self.num_fits < len(self.param_displays[0]):
            for i in range(5):
                self.param_displays[i][-1].setText('')
                self.param_displays[i].pop(-1)
        for i in range(5):
            for j in range(len(self.param_displays[i])):
                if update_vals:
                    if 2 <= i <= 3:
                        if self.num_fits > 1:
                            self.param_displays[i][j].setText('%.1f' % (self.p[j][i] * 2 * self.unit))
                        else:
                            self.param_displays[i][j].setText('%.1f' % (self.p[i] * 2 * self.unit))
                    else:
                        if self.num_fits > 1:
                            self.param_displays[i][j].setText('%.1f' % self.p[j][i])
                        else:
                            self.param_displays[i][j].setText('%.1f' % self.p[i])
                else:
                    self.param_displays[i][j].setText('N/A')

    def zoom(self):
        self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
        if self.target_scale <= 0:
            self.loclist = []
            self.xlist = []
            self.ylist = []
            self.target_scale = 0
            self.current_scale = 0

        else:
            if self.target_scale > self.current_scale:
                self.loclist.append([self.zoom_x, self.zoom_y])
            elif self.target_scale < self.current_scale:
                self.loclist.pop()
            self.current_scale = 0
            for i in self.loclist:
                self.current_scale += 1
                imgsize = (self.pixmap.width(), self.pixmap.height())
                [ix, iy] = i
                ix = float(ix) * self.pixmap.width() / float(self.cam_controller.framewidth)
                iy = float(iy) * self.pixmap.height() / float(self.cam_controller.frameheight)

                if ix < round(imgsize[0]/4):
                    xcorner = 0
                elif ix > round(3*imgsize[0]/4):
                    xcorner = round(imgsize[0]/2)
                else:
                    xcorner = round(ix - imgsize[0]/4)
                if iy < round(imgsize[1]/4):
                    ycorner = 0
                elif iy > round(3*imgsize[1]/4):
                    ycorner = round(imgsize[1]/2)
                else:
                    ycorner = round(iy - imgsize[1]/4)
                rect = QRect(xcorner,ycorner,2*imgsize[0]/4,2*imgsize[1]/4)

                #xoffset=min(ix,imgsize[0]-ix,imgsize[0]/4)                    #reduces the image to half or takes the boundary closest to the mouse point
                #yoffset = min(iy, imgsize[1] - iy, imgsize[1] / 4)             #change (1/4) to change the zoom step size
                #xoffset1=min(float(xoffset)/float(imgsize[0]),float(yoffset)/float(imgsize[1]))*imgsize[0]
                #yoffset1 = min(float(xoffset)/float(imgsize[0]), float(yoffset)/float(imgsize[1])) * imgsize[1]
                #rect = QRect(ix-xoffset1, iy-yoffset1, 2*xoffset1, 2*yoffset1)
                self.xlist.append(xcorner)
                self.ylist.append(ycorner)
                self.pixmap = self.pixmap.copy(rect)
                #self.xbox.append(self.pixmap.width())
                #self.ybox.append(self.pixmap.height())
                self.rect.setCoords(self.rect.x() + rect.x(), self.rect.y() + rect.y(), self.rect.x() + rect.right(),
                                    self.rect.y() + rect.bottom())
        self.plotx.setXRange(self.rect.left(), self.rect.right())
        self.ploty.setYRange(self.rect.top(), self.rect.bottom())

    def update_plot_withoutfit(self):
        stime=time.time()
        self.section_xcoord = np.arange(0, self.cam_controller.frame.shape[1])
        self.section_xdata = self.cam_controller.frame[self.section_yctr,::]
        self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
        self.section_ycoord = np.arange(0, self.cam_controller.frame.shape[0])
        self.section_ydata = self.cam_controller.frame[::,self.section_xctr]
        self.sectiony_line.setData(self.section_ydata,self.section_ycoord)

        self.section_xdata_fit = []
        self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
        self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
        for i in range(len(self.sectionx_fit_mult)):
            self.sectionx_fit_mult[i].setData(self.section_xcoord, [])
        self.section_ydata_fit = []
        self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
        self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]
        for i in range(len(self.sectiony_fit_mult)):
            self.sectiony_fit_mult[i].setData(self.section_ydata_fit_mult[i], self.section_ycoord)
        print('6:' + str((time.time() - stime) * 1000))

    def update_plot(self):
        # self.line_edit_multi_fits()
        # self.line_edit_slices()
        # stime=time.time()
        self.section_xcoord = np.arange(0, self.cam_controller.frame.shape[1])
        self.section_xdata = self.cam_controller.frame[self.section_yctr,::]
        self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
        self.section_ycoord = np.arange(0, self.cam_controller.frame.shape[0])
        self.section_ydata = self.cam_controller.frame[::,self.section_xctr]
        self.sectiony_line.setData(self.section_ydata,self.section_ycoord)

        # print('test2')

        if self.num_fits <= 1 or not isinstance(self.p[0], list):
            # Create the lines for the fitted graph to be plotted below the data
            self.section_xdata_fit = self.gauss_data(self.p[4], self.p[0], self.p[2], self.p[5], self.section_xcoord.shape[0])
            self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
            self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
            for i in range(len(self.sectionx_fit_mult)):
                self.sectionx_fit_mult[i].setData([], [])

            self.section_ydata_fit = self.gauss_data(self.p[4], self.p[1], self.p[3], self.p[5], self.section_ycoord.shape[0])
            self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
            self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]
            for i in range(len(self.sectiony_fit_mult)):
                self.sectiony_fit_mult[i].setData([], [])
        else:
            self.section_xdata_fit = []
            self.sectionx_fit.setData([], [])
            self.section_ydata_fit = []
            self.sectiony_fit.setData([], [])
            self.section_xdata_fit_mult = [[] for _ in range(len(self.p))]
            for i in range(len(self.sectionx_fit_mult)):
                self.sectionx_fit_mult[i].setData([], [])
            self.section_ydata_fit_mult = [[] for _ in range(len(self.p))]
            for i in range(len(self.sectiony_fit_mult)):
                self.sectiony_fit_mult[i].setData([], [])
            for fit_num in range(len(self.p)):
                # print('test3')
                # self.section_xdata_fit = (self.gauss_data(fit[4], fit[0], fit[2], fit[5], self.section_xcoord.shape[0]))
                # self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
                # self.plotx.plot(self.section_xcoord, self.section_xdata_fit,
                #                 pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
                self.section_xdata_fit_mult[fit_num] = self.gauss_data(self.p[fit_num][4], self.p[fit_num][0],
                                                                       self.p[fit_num][2], self.p[fit_num][5],
                                                                       self.section_xcoord.shape[0])
                if len(self.colors) <= fit_num:
                    self.colors.append((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
                if len(self.sectionx_fit_mult) <= fit_num:
                    self.sectionx_fit_mult.append(self.plotx.plot(self.section_xcoord,
                                                                  self.section_xdata_fit_mult[fit_num],
                                                  pen=mkPen(color=self.colors[-1], style=QtCore.Qt.DashLine)))
                else:
                    self.sectionx_fit_mult[fit_num].setData(self.section_xcoord, self.section_xdata_fit_mult[fit_num])

                # self.section_ydata_fit = (self.gauss_data(fit[4], fit[1], fit[3], fit[5], self.section_ycoord.shape[0]))
                # self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
                # self.ploty.plot(self.section_ydata_fit, self.section_ycoord,
                #                 pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
                self.section_ydata_fit_mult[fit_num] = self.gauss_data(self.p[fit_num][4], self.p[fit_num][1],
                                                                       self.p[fit_num][3], self.p[fit_num][5],
                                                                       self.section_ycoord.shape[0])
                if len(self.sectiony_fit_mult) <= fit_num:
                    self.sectiony_fit_mult.append(self.ploty.plot(self.section_ydata_fit_mult[fit_num],
                                                                  self.section_ycoord,
                                                                  pen=mkPen(color=self.colors[-1],
                                                                            style=QtCore.Qt.DashLine)))
                else:
                    self.sectiony_fit_mult[fit_num].setData(self.section_ydata_fit_mult[fit_num], self.section_ycoord)

        # center logging
        with open('center_log/' + self.start_date + '.txt', 'a') as f:
            if self.num_fits <= 1 or not isinstance(self.p[0], list):
                f.write(str(time.time() - self.start_time) + ', ' + str(self.p[0]) + ', ' + str(self.p[1]) + '\n')
            else:
                f.write(str(time.time() - self.start_time))
                for fit in self.p:
                    f.write(', ' + str(fit[0]) + ', ' + str(fit[1]))
                f.write('\n')
        # print('7:' + str((time.time() - stime) * 1000))

    # Gaussian function
    def gauss(self, t, A, mu, sigma, offset):
        return A * (math.e ** (-1 * ((t - mu) ** 2) / (2 * (sigma ** 2)))) + offset

    # Plot a gaussian over a range [0, data_range)
    def gauss_data(self, A, mu, sigma, offset, data_range):
        out = []
        for i in range(data_range):
            out.append(self.gauss(i, A, mu, sigma, offset))
        return out

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

    def draw_crosshair(self,mouse_x,mouse_y):
        label_width = self.labelImage.size().width()
        label_height = self.labelImage.size().height()
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen()
        pen.setWidth(5) #reduced width of crosshair from 20 to 5
        pen.setColor(QtGui.QColor('green'))
        painter.setPen(pen)
        painter.setOpacity(0.4)
        painter.drawLine((mouse_x*self.pixmap.width())/label_width,0,(mouse_x*self.pixmap.width())/label_width,self.pixmap.height())
        painter.drawLine(0,(mouse_y*self.pixmap.height())/label_height,self.pixmap.width(),(mouse_y*self.pixmap.height())/label_height)
        painter.end()

    def set_exptime(self):
        self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
        self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))

    def label_mousepress(self):

        def mousepress(eventQMouseEvent):
            label_width = self.labelImage.size().width()
            label_height = self.labelImage.size().height()
            self.mouse_x = eventQMouseEvent.pos().x()
            self.mouse_y = eventQMouseEvent.pos().y()
            if len(self.loclist) == 0:
                self.section_xctr = round(self.cam_controller.framewidth * self.mouse_x / label_width)
                self.section_yctr = round(self.cam_controller.frameheight * self.mouse_y / label_height)
            else:
                self.section_xctr = round(self.pixmap.width()* self.mouse_x / label_width)
                self.section_yctr = round(self.pixmap.height()*self.mouse_y/label_height)
                for i in range(len(self.loclist)):
                    self.section_xctr += self.xlist[len(self.xlist)-i-1]
                    self.section_yctr += self.ylist[len(self.ylist)-i-1]
            self.lineEditSectionX.setText(str(self.section_xctr))
            self.lineEditSectionY.setText(str(self.section_yctr))


        # def mousepress(eventQMouseEvent):
        #     label_width = self.labelImage.size().width()
        #     label_height = self.labelImage.size().height()
        #     self.mouse_x = eventQMouseEvent.pos().x()
        #     self.mouse_y = eventQMouseEvent.pos().y()
        #     self.section_xctr = round(self.cam_controller.framewidth * self.mouse_x / label_width)
        #     self.section_yctr = round(self.cam_controller.frameheight*self.mouse_y/label_height)
        #
        #     if len(self.loclist) == 0:
        #         self.lineEditSectionX.setText(str(self.section_xctr))
        #         self.lineEditSectionY.setText(str(self.section_yctr))
        #     else:
        #         self.section_xctr_pseudo = round(self.pixmap.width()* self.mouse_x / label_width)
        #         self.section_yctr_pseudo = round(self.pixmap.height()*self.mouse_y/label_height)
        #         for i in range(len(self.loclist)):
        #                 self.section_xctr_pseudo += self.xlist[len(self.xlist)-i-1]
        #                 self.section_yctr_pseudo += self.ylist[len(self.ylist)-i-1]
        #         self.lineEditSectionX.setText(str(self.section_xctr_pseudo))
        #         self.lineEditSectionY.setText(str(self.section_yctr_pseudo))


            # if len(self.loclist) == 0:
            #     self.section_xctr = round(self.cam_controller.framewidth*self.mouse_x/label_width)
            #     self.section_yctr = round(self.cam_controller.frameheight*self.mouse_y/label_height)
            # else:
            #     self.section_xctr = round(self.pixmap.width()*self.mouse_x/label_width)
            #     self.section_yctr = round(self.pixmap.height()*self.mouse_y/label_height)
            #     for i in range(len(self.loclist)):
            #         self.section_xctr += self.xlist[i]
            #         self.section_yctr += self.ylist[i]
            #         print(str(self.xlist[i]))
            #         print(str(self.ylist[i]))
            # self.lineEditSectionX.setText(str(self.section_xctr))
            # self.lineEditSectionY.setText(str(self.section_yctr))


            if self.checkBoxFit.isChecked():
                self.update_plot()
            else:
                self.update_plot_withoutfit()
        return mousepress

    def label_mousewheel(self):
        def mousewheel(event):
            if self.checkBoxZoom.isChecked():
                screenpoint = self.labelImage.mapFromGlobal(QtGui.QCursor.pos())
                self.zoom_x, self.zoom_y = event.pos().x() + screenpoint.x(), event.pos().y() + screenpoint.y()
                self.zoom_x = float(self.zoom_x)*self.cam_controller.framewidth/(2*self.labelImage.size().width())  # scaling from mouse position to display position
                self.zoom_y = float(self.zoom_y)*self.cam_controller.frameheight/(2*self.labelImage.size().height())
                if event.angleDelta().y() > 0:
                    if self.target_scale < 5:
                        self.target_scale += 1
                    else:
                        print('Reached Zoom Limit!')
                else:
                    self.target_scale -= 1
                self.update_movie()
            else:
                self.update_movie()
        return mousewheel

    def section_center(self):
        # self.section_xctr = round(float(self.lineEditxCenter.text())/self.unit)
        # self.section_yctr = round(float(self.lineEdityCenter.text())/self.unit)

        # Instead change center to the centers of each gaussian
        if self.num_fits <= 1:
            self.section_xctr = round(self.p[0])
            self.section_yctr = round(self.p[1])
        else:
            self.section_xctr = round(self.p[0][0])
            self.section_yctr = round(self.p[0][1])

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

    def line_edit_multi_fits(self):
        try:
            if int(self.lineEditMultiFits.text()) > 0:
                self.num_fits = int(self.lineEditMultiFits.text())
            else:
                self.num_fits = 1
        except ValueError:
            self.plainTextEditLog.insertPlainText('Please enter an integer value for the number of fits\n')

    # def line_edit_slices(self):
    #     try:
    #         if int(self.lineEditSlices.text()) > 0:
    #             self.slices = int(self.lineEditSlices.text())
    #         else:
    #             self.slices = 40
    #     except ValueError:
    #         self.plainTextEditLog.insertPlainText('Please enter an integer value for the number of slices\n')

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
        name = QtWidgets.QFileDialog.getSaveFileName(self.mainwindow, 'Save File', os.path.join(self.save_dir, filename),
                                                 "Images (*.png *.jpg)")[0]
        # name = QtGui.QFileDialog.getSaveFileName(self.mainwindow, 'Save File',os.path.join(self.save_dir,filename),"Images (*.png *.jpg)")[0]
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