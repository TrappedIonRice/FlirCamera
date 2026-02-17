import time
import os
import datetime
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import QRect, QPoint
from pyqtgraph import PlotWidget, mkPen

from FlirWindow import Ui_MainWindow
from fitgauss import fitgauss2d_section, fitgauss2d_multiple


# ==================================================================================
# CLASS: SmartViewer
# Purpose: A custom QLabel subclass that handles:
#   1. Displaying the camera image with CORRECT Aspect Ratio.
#   2. "Sharp" rendering (Nearest Neighbor).
#   3. Drawing a "Crosshair" overlay.
#   4. Prevents infinite window resizing loops.
# ==================================================================================
class SmartViewer(QtWidgets.QLabel):
    def __init__(self, parent=None):
        """Initializes the viewer widget settings."""
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self._pixmap = None
        self.crosshair_enabled = True
        self.crosshair_pos = None
        self.draw_rect = QRect()

    def set_image(self, pixmap):
        """Updates the internal pixmap and triggers a repaint."""
        self._pixmap = pixmap
        self.update()

    def set_crosshair(self, enabled, pos=None):
        """Updates crosshair visibility and position."""
        self.crosshair_enabled = enabled
        if pos:
            self.crosshair_pos = pos
        self.update()

    def paintEvent(self, event):
        """Custom painting to handle Aspect Ratio and Crosshair overlay."""
        painter = QPainter(self)
        bg_color = self.palette().color(QtGui.QPalette.Window)
        painter.fillRect(self.rect(), bg_color)

        if self._pixmap and not self._pixmap.isNull():
            w_widget, h_widget = self.width(), self.height()
            w_img, h_img = self._pixmap.width(), self._pixmap.height()

            if w_img > 0 and h_img > 0:
                scale = min(w_widget / w_img, h_widget / h_img)
            else:
                scale = 1

            draw_w = int(w_img * scale)
            draw_h = int(h_img * scale)
            off_x = (w_widget - draw_w) // 2
            off_y = (h_widget - draw_h) // 2

            self.draw_rect = QRect(off_x, off_y, draw_w, draw_h)
            target_rect = self.draw_rect
            source_rect = QRect(0, 0, w_img, h_img)

            painter.drawPixmap(target_rect, self._pixmap, source_rect)

            if self.crosshair_enabled and self.crosshair_pos:
                pen = QPen(QtCore.Qt.green)
                pen.setWidth(1)
                pen.setCosmetic(True)
                painter.setPen(pen)

                pm_x, pm_y = self.crosshair_pos
                screen_x = off_x + (pm_x * scale)
                screen_y = off_y + (pm_y * scale)
                ix, iy = int(screen_x), int(screen_y)

                img_top, img_bottom = off_y, off_y + draw_h
                img_left, img_right = off_x, off_x + draw_w

                if (img_left <= ix <= img_right) and (img_top <= iy <= img_bottom):
                    painter.drawLine(ix, img_top, ix, img_bottom)
                    painter.drawLine(img_left, iy, img_right, iy)
        painter.end()


# ==================================================================================
# CLASS: Ui_CustomWindow
# Purpose: Main application logic.
# ==================================================================================
class Ui_CustomWindow(Ui_MainWindow):

    def custom_init(self, mainwindow):
        """Initializes the GUI, Camera, Variables, and Connections."""
        self.mainwindow = mainwindow

        # --- 1. SETUP SMART VIEWER ---
        self.old_label = self.labelImage
        self.labelImage = SmartViewer(self.centralwidget)
        self.labelImage.setSizePolicy(self.old_label.sizePolicy())
        self.gridLayoutImage.replaceWidget(self.old_label, self.labelImage)
        self.old_label.deleteLater()

        # --- 2. INITIALIZE VARIABLES ---
        self.stime = time.time()
        self.nframe = 0
        self.section_xctr = 0
        self.section_yctr = 0
        self.section_xdata = []
        self.section_ydata = []
        self.section_xcoord = []
        self.section_ycoord = []
        self.unit = 1.0
        self.iter = -1

        # Zoom & Pan Variables
        self.loclist = []
        self.zoom_x = 0
        self.zoom_y = 0
        self.current_scale = 0
        self.target_scale = 0
        self.pan_x = 0
        self.pan_y = 0
        self.last_drag_pos = QtCore.QPoint()
        self.mouse_x = 0
        self.mouse_y = 0
        self.zoom_needed = False

        # Logging Variables
        self.current_log_session = None
        self.log_start_time = 0.0
        self.avg_win_width = 50
        self.data_buffer_x = []
        self.data_buffer_y = []

        # Multi-Fit Variables
        self.num_fits = 1
        self.slices = 40
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        self.multi_curves_x = []
        self.multi_curves_y = []

        self.sectionx_fit_mult = []
        self.sectiony_fit_mult = []

        # Dynamic Display Layouts (List of 5 Rows)
        self.param_displays = [[] for _ in range(5)]
        self.param_layout = QtWidgets.QGridLayout()
        self.param_layout.setObjectName('gridLayoutParameters')
        self.param_label_layout = QtWidgets.QGridLayout()
        self.param_label_layout.setObjectName('gridLayoutParameterLabels')

        self.param_font = QFont("Arial", 20)
        self.param_font.setBold(True)

        # --- 3. CAMERA INITIALIZATION ---
        self.cam_controller.log = self.plainTextEditLog
        self.cam_controller.initialize()

        # Set Default Exposure
        self.cam_controller.configure_exposure(89.0)

        fw = getattr(self.cam_controller, 'framewidth', 1280)
        fh = getattr(self.cam_controller, 'frameheight', 1024)
        self.rect = QRect(0, 0, fw, fh)
        self.section_xctr = fw // 2
        self.section_yctr = fh // 2

        # --- 4. UI CONNECTIONS ---
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_movie)
        self.pushButtonContinue.clicked.connect(self.start_continue)

        self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
        self.lineEditExposureTime.returnPressed.connect(self.set_exptime)

        # Plots
        self.plotx = PlotWidget(self.centralwidget)
        self.gridLayoutImage.addWidget(self.plotx, 1, 0, 1, 1)
        self.ploty = PlotWidget(self.centralwidget)
        self.gridLayoutImage.addWidget(self.ploty, 0, 1, 1, 1)
        self.ploty.getPlotItem().getViewBox().invertY(True)

        self.sectionx_line = self.plotx.plot([], [])
        self.sectiony_line = self.ploty.plot([], [])
        self.sectionx_fit = self.plotx.plot([], [], pen=mkPen(color=self.colors[0], style=QtCore.Qt.DashLine))
        self.sectiony_fit = self.ploty.plot([], [], pen=mkPen(color=self.colors[0], style=QtCore.Qt.DashLine))

        self.labelImage.mousePressEvent = self.label_mousepress()
        self.labelImage.mouseMoveEvent = self.label_mousemove()
        self.labelImage.wheelEvent = self.label_mousewheel()

        self.pushButtonSectionCenter.clicked.connect(self.section_center)
        self.lineEditAverageFrames.returnPressed.connect(self.set_average_frames)
        self.radioButtonUnitUm.toggled.connect(self.unit_change)
        self.radioButtonUnitPixel.toggled.connect(self.unit_change)
        self.checkBoxAutoExposure.stateChanged.connect(self.checkbox_auto_exposure)

        # Default Fit Count = 1
        self.lineEditMultiFits.setText("1")
        self.lineEditMultiFits.returnPressed.connect(self.line_edit_multi_fits)

        self.pushButtonSetBg.clicked.connect(self.set_background)
        self.pushButtonClearBg.clicked.connect(self.clear_background)
        self.checkBoxLogData.stateChanged.connect(self.handle_logging_toggle)
        self.lineEditLogWindow.returnPressed.connect(self.set_log_window)

        if hasattr(self, 'actionsave'):
            self.actionsave.triggered.connect(self.file_save)
        if hasattr(self, 'actionsave_image'):
            self.actionsave_image.triggered.connect(self.file_save)
        if hasattr(self, 'actionsave_image_2'):
            self.actionsave_image_2.triggered.connect(self.file_save)

        mainwindow.setWindowIcon(QtGui.QIcon('logo.png'))
        self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
        self.start_date = str(datetime.datetime.now()).replace(':', '-')
        self.start_time = time.time()

        if not os.path.exists('log_t_x_y_wx_wy_h'): os.makedirs('log_t_x_y_wx_wy_h')
        if not os.path.exists('logged_images_auto'): os.makedirs('logged_images_auto')

        self.unit_change()
        self.set_log_window()

    def start_continue(self):
        """Starts the video loop."""
        self.cam_controller.start_continue()
        self.pushButtonContinue.setText("Stop")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.stop_continue)
        self.update_timer.start(30)
        self.iter = 0
        self.stime = time.time()

    def stop_continue(self):
        """Stops the video loop."""
        self.update_timer.stop()
        self.cam_controller.stop_continue()
        self.pushButtonContinue.setText("Start")
        self.pushButtonContinue.clicked.disconnect()
        self.pushButtonContinue.clicked.connect(self.start_continue)

    def update_movie(self):
        """Main Loop: Acquires frame, Fits Data, Updates UI & Plots."""
        self.cam_controller.acquire_continue()
        frame = self.cam_controller.frame
        if frame is None or frame.size == 0:
            return

        if hasattr(self.cam_controller, 'framewidth'):
            if self.cam_controller.framewidth != frame.shape[1]:
                self.cam_controller.framewidth = frame.shape[1]
                self.cam_controller.frameheight = frame.shape[0]
                if self.rect.width() != frame.shape[1] and self.target_scale == 0:
                    self.rect = QRect(0, 0, frame.shape[1], frame.shape[0])

        if self.checkBoxFit.isChecked():
            # --- SINGLE FIT MODE ---
            if self.num_fits <= 1:
                self.setup_multi_fit_display(1)

                if not self.zoom_needed:
                    p, ier = fitgauss2d_section(np.arange(0, frame.shape[1]), np.arange(0, frame.shape[0]), frame)
                else:
                    p, ier = fitgauss2d_section(np.arange(0, frame.shape[1]), np.arange(0, frame.shape[0]), frame)

                self.p = [p]

                # Stats Buffer
                self.data_buffer_x.append(p[0])
                self.data_buffer_y.append(p[1])
                if len(self.data_buffer_x) > self.avg_win_width:
                    self.data_buffer_x.pop(0)
                    self.data_buffer_y.pop(0)

                x_avg = np.mean(self.data_buffer_x)
                y_avg = np.mean(self.data_buffer_y)
                x_std = np.std(self.data_buffer_x)
                y_std = np.std(self.data_buffer_y)

                if self.checkBoxLogData.isChecked() and self.current_log_session:
                    self.log_data_to_file(p, frame)

                # Standard Display
                if self.lineEditxCenter.parent() is not None:
                    self.lineEditxCenter.setText('%.2f' % (p[0] * self.unit))
                    self.lineEditAVGxCenter.setText('%.2f' % (x_avg * self.unit))
                    self.lineEditDxCenter.setText('%.2f' % (x_std * self.unit))
                    self.lineEdityCenter.setText('%.2f' % (p[1] * self.unit))
                    self.lineEditAVGyCenter.setText('%.2f' % (y_avg * self.unit))
                    self.lineEditDyCenter.setText('%.2f' % (y_std * self.unit))
                    self.lineEditxWaist.setText('%.1f' % (p[2] * 2 * self.unit))
                    self.lineEdityWaist.setText('%.1f' % (p[3] * 2 * self.unit))
                    self.lineEditHeight.setText('%.1f' % p[4])

                self.update_fit_plots(p, frame, is_multi=False)

            # --- MULTI FIT MODE ---
            else:
                self.setup_multi_fit_display(self.num_fits)

                if not self.zoom_needed:
                    self.p = fitgauss2d_multiple(frame, np.arange(frame.shape[0]), np.arange(frame.shape[1]),
                                                 num_fits=self.num_fits, slices=self.slices)
                else:
                    self.p = fitgauss2d_multiple(frame, np.arange(frame.shape[0]), np.arange(frame.shape[1]),
                                                 num_fits=self.num_fits, slices=self.slices)

                self.update_raw_plots(frame)

                # Swap UI if needed
                if self.lineEditxCenter.parent() is not None:
                    self.switch_to_multi_mode()

                self.update_param_displays()

                # Plotting curves
                def gauss(x, A, mu, sigma, offset):
                    if sigma == 0: sigma = 0.001
                    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

                for i, fit in enumerate(self.p):
                    val_x, val_y = fit[1], fit[0]
                    val_wx, val_wy = fit[3], fit[2]
                    val_h, val_f = fit[4], fit[5]

                    fx = gauss(self.section_xcoord, val_h, val_x, val_wx, val_f)
                    self.sectionx_fit_mult[i].setData(self.section_xcoord, fx)
                    fy = gauss(self.section_ycoord, val_h, val_y, val_wy, val_f)
                    self.sectiony_fit_mult[i].setData(fy, self.section_ycoord)

                    if i == 0 and self.checkBoxLogData.isChecked() and self.current_log_session:
                        p_std = [val_x, val_y, val_wx, val_wy, val_h, val_f]
                        self.log_data_to_file(p_std, frame)
        else:
            self.update_raw_plots(frame)
            self.lineEditAVGxCenter.setText('N/A')
            self.lineEditAVGyCenter.setText('N/A')

        self.iter += 1
        self.pixmap = QtGui.QPixmap(self.toQImage())
        self.zoom()

        if self.section_xctr != 0 and self.section_yctr != 0 and self.checkBoxCrosshair.isChecked():
            self.draw_crosshair()
        else:
            self.labelImage.set_crosshair(False)

        self.labelImage.set_image(self.pixmap)
        if self.checkBoxAutoExposure.isChecked():
            self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))

    def switch_to_multi_mode(self):
        """Swaps standard widgets for dynamic grid."""
        self.gridLayoutFitResult.addLayout(self.param_label_layout, 0, 0, 6, 1)
        self.gridLayoutFitResult.addLayout(self.param_layout, 0, 1, 6, 4)

        self.lineEditxCenter.setParent(None)
        self.lineEditAVGxCenter.setParent(None);
        self.lineEditDxCenter.setParent(None)
        self.lineEdityCenter.setParent(None)
        self.lineEditAVGyCenter.setParent(None);
        self.lineEditDyCenter.setParent(None)
        self.lineEditxWaist.setParent(None);
        self.lineEdityWaist.setParent(None)
        self.lineEditHeight.setParent(None)

        self.labelAVG.setParent(None)
        self.labelD.setParent(None)

        self.labelxCenter.setParent(None);
        self.param_label_layout.addWidget(self.labelxCenter, 1, 0, 1, 1)
        self.labelyCenter.setParent(None);
        self.param_label_layout.addWidget(self.labelyCenter, 2, 0, 1, 1)
        self.labelxWaist.setParent(None);
        self.param_label_layout.addWidget(self.labelxWaist, 3, 0, 1, 1)
        self.labelyWaist.setParent(None);
        self.param_label_layout.addWidget(self.labelyWaist, 4, 0, 1, 1)
        self.labelHeight.setParent(None);
        self.param_label_layout.addWidget(self.labelHeight, 5, 0, 1, 1)

    def switch_to_single_mode(self):
        """Restores standard widgets."""
        self.gridLayoutFitResult.removeItem(self.param_label_layout)
        self.gridLayoutFitResult.removeItem(self.param_layout)

        self.labelxCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.labelxCenter, 1, 0, 1, 1)
        self.labelyCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.labelyCenter, 2, 0, 1, 1)
        self.labelxWaist.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.labelxWaist, 3, 0, 1, 1)
        self.labelyWaist.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.labelyWaist, 4, 0, 1, 1)
        self.labelHeight.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.labelHeight, 5, 0, 1, 1)

        self.labelAVG.setParent(self.centralwidget);
        self.labelAVG.setVisible(True)
        self.gridLayoutFitResult.addWidget(self.labelAVG, 0, 2, 1, 1)
        self.labelD.setParent(self.centralwidget);
        self.labelD.setVisible(True)
        self.gridLayoutFitResult.addWidget(self.labelD, 0, 3, 1, 1)

        self.lineEditxCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditxCenter, 1, 1, 1, 1)
        self.lineEditAVGxCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditAVGxCenter, 1, 2, 1, 1)
        self.lineEditDxCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditDxCenter, 1, 3, 1, 1)
        self.lineEdityCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEdityCenter, 2, 1, 1, 1)
        self.lineEditAVGyCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditAVGyCenter, 2, 2, 1, 1)
        self.lineEditDyCenter.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditDyCenter, 2, 3, 1, 1)
        self.lineEditxWaist.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditxWaist, 3, 1, 1, 3)
        self.lineEdityWaist.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEdityWaist, 4, 1, 1, 3)
        self.lineEditHeight.setParent(self.centralwidget);
        self.gridLayoutFitResult.addWidget(self.lineEditHeight, 5, 1, 1, 3)

    def update_param_displays(self):
        """Updates dynamic text labels for multi-fit."""
        # Create labels for all beams (Including Beam 1) if missing
        while self.num_fits > len(self.param_displays[0]):
            for i in range(5):
                lbl = QtWidgets.QLabel(self.centralwidget)
                lbl.setFont(self.param_font)
                idx = len(self.param_displays[i])
                col_c = self.colors[idx % len(self.colors)]
                style = f"color: rgb({col_c[0]}, {col_c[1]}, {col_c[2]}); background-color: black"
                lbl.setStyleSheet(style)
                lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.param_displays[i].append(lbl)
                self.param_layout.addWidget(lbl, i, idx + 1, 1, 1)

        # Remove extra labels
        while self.num_fits < len(self.param_displays[0]):
            for i in range(5):
                lbl = self.param_displays[i].pop()
                lbl.setParent(None);
                lbl.deleteLater()

        # Update Text
        for j in range(len(self.p)):
            fit = self.p[j]
            vals = [
                fit[1] * self.unit,  # X
                fit[0] * self.unit,  # Y
                fit[3] * np.sqrt(2) * self.unit,  # waist is sqrt(2) * sigma
                fit[2] * np.sqrt(2) * self.unit,  # Wy
                fit[4]  # H
            ]
            for i in range(5):
                if j < len(self.param_displays[i]):
                    self.param_displays[i][j].setText('%.1f' % vals[i])

    def setup_multi_fit_display(self, n_fits):
        """Manages curves for multi-fit display."""
        if n_fits <= 1:
            # Clear Multi Curves
            for c in self.sectionx_fit_mult: self.plotx.removeItem(c)
            for c in self.sectiony_fit_mult: self.ploty.removeItem(c)
            self.sectionx_fit_mult = []
            self.sectiony_fit_mult = []

            # Clear Dynamic Labels
            while len(self.param_displays[0]) > 0:
                for i in range(5):
                    lbl = self.param_displays[i].pop()
                    lbl.setParent(None);
                    lbl.deleteLater()

        else:
            # Create curves for ALL beams (0 to N-1)
            while len(self.sectionx_fit_mult) < n_fits:
                col = self.colors[len(self.sectionx_fit_mult) % len(self.colors)]
                self.sectionx_fit_mult.append(self.plotx.plot([], [], pen=mkPen(color=col, style=QtCore.Qt.DashLine)))
                self.sectiony_fit_mult.append(self.ploty.plot([], [], pen=mkPen(color=col, style=QtCore.Qt.DashLine)))

            # Trim excess curves
            while len(self.sectionx_fit_mult) > n_fits:
                self.plotx.removeItem(self.sectionx_fit_mult.pop())
                self.ploty.removeItem(self.sectiony_fit_mult.pop())

    def log_data_to_file(self, p, frame):
        """Logs (Time, X, Y, Wx, Wy, H) to text file."""
        elapsed_time = time.time() - self.log_start_time
        log_path = 'log_t_x_y_wx_wy_h/' + self.current_log_session + '.txt'
        with open(log_path, 'a') as f:
            f.write(f"{elapsed_time:.4f} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]}\n")

    def plot_gaussian(self, curve_x, curve_y, p):
        def gauss(x, A, mu, sigma, offset):
            if sigma == 0: sigma = 0.001
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

        fit_x = gauss(self.section_xcoord, p[4], p[0], p[2], p[5])
        curve_x.setData(self.section_xcoord, fit_x)
        fit_y = gauss(self.section_ycoord, p[4], p[1], p[3], p[5])
        curve_y.setData(fit_y, self.section_ycoord)

    def update_raw_plots(self, frame):
        self.section_xcoord = np.arange(0, frame.shape[1])
        self.section_xdata = frame[self.section_yctr, :]
        self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
        self.section_ycoord = np.arange(0, frame.shape[0])
        self.section_ydata = frame[:, self.section_xctr]
        self.sectiony_line.setData(self.section_ydata, self.section_ycoord)

    def update_fit_plots(self, p, frame, is_multi=False):
        self.update_raw_plots(frame)
        if not is_multi:
            self.plot_gaussian(self.sectionx_fit, self.sectiony_fit, p)
        else:
            self.sectionx_fit.setData([], [])
            self.sectiony_fit.setData([], [])

    def zoom(self):
        # 1. Reset Rect to Full Frame
        fw = getattr(self.cam_controller, 'framewidth', 1280)
        fh = getattr(self.cam_controller, 'frameheight', 1024)
        self.rect = QRect(0, 0, fw, fh)

        if self.target_scale <= 0:
            self.loclist = []
            self.target_scale = 0
            self.current_scale = 0
            self.pan_x = 0
            self.pan_y = 0
        else:
            if self.target_scale > self.current_scale:
                self.loclist.append([self.zoom_x, self.zoom_y])
            elif self.target_scale < self.current_scale:
                if self.loclist: self.loclist.pop()

            self.current_scale = 0
            ZOOM_FACTOR = 0.8

            for i in self.loclist:
                self.current_scale += 1
                imgsize = (self.pixmap.width(), self.pixmap.height())
                [ix, iy] = i

                # Map Global Mouse -> Local Pixel in current view
                ix_curr = ix - self.rect.left()
                iy_curr = iy - self.rect.top()

                new_w = int(imgsize[0] * ZOOM_FACTOR)
                new_h = int(imgsize[1] * ZOOM_FACTOR)

                xcorner = max(0, min(int(ix_curr - new_w / 2), imgsize[0] - new_w))
                ycorner = max(0, min(int(iy_curr - new_h / 2), imgsize[1] - new_h))

                rect = QRect(xcorner, ycorner, new_w, new_h)
                self.pixmap = self.pixmap.copy(rect)
                self.rect.setCoords(
                    self.rect.x() + rect.x(),
                    self.rect.y() + rect.y(),
                    self.rect.x() + rect.right(),
                    self.rect.y() + rect.bottom()
                )

        if self.pan_x != 0 or self.pan_y != 0:
            shifted_rect = self.rect.translated(self.pan_x, self.pan_y)
            if shifted_rect.left() < 0: shifted_rect.moveLeft(0)
            if shifted_rect.top() < 0: shifted_rect.moveTop(0)
            if shifted_rect.right() > fw: shifted_rect.moveRight(fw)
            if shifted_rect.bottom() > fh: shifted_rect.moveBottom(fh)
            self.rect = shifted_rect
            raw_qimg = self.toQImage(copy=False)
            self.pixmap = QtGui.QPixmap.fromImage(raw_qimg).copy(self.rect)

        self.plotx.setXRange(self.rect.left(), self.rect.right())
        self.ploty.setYRange(self.rect.top(), self.rect.bottom())

    def draw_crosshair(self):
        local_x = self.section_xctr - self.rect.left()
        local_y = self.section_yctr - self.rect.top()
        if self.rect.width() > 0 and self.rect.height() > 0:
            pm_x = local_x * (self.pixmap.width() / self.rect.width())
            pm_y = local_y * (self.pixmap.height() / self.rect.height())
            self.labelImage.set_crosshair(True, (pm_x, pm_y))
        else:
            self.labelImage.set_crosshair(False)

    def label_mousepress(self):
        def mousepress(event):
            if event.button() == QtCore.Qt.RightButton:
                self.last_drag_pos = event.pos()
                return
            if event.button() == QtCore.Qt.LeftButton:
                label_w = self.labelImage.width()
                label_h = self.labelImage.height()
                pix_w = self.pixmap.width()
                pix_h = self.pixmap.height()
                if pix_w == 0 or pix_h == 0: return
                scale = min(label_w / pix_w, label_h / pix_h)
                draw_w = int(pix_w * scale)
                draw_h = int(pix_h * scale)
                off_x = (label_w - draw_w) // 2
                off_y = (label_h - draw_h) // 2
                click_x = event.pos().x() - off_x
                click_y = event.pos().y() - off_y
                if click_x < 0 or click_x > draw_w or click_y < 0 or click_y > draw_h: return
                pm_x = click_x / scale
                pm_y = click_y / scale

                fw = getattr(self.cam_controller, 'framewidth', 1280)
                fh = getattr(self.cam_controller, 'frameheight', 1024)

                global_x = self.rect.left() + pm_x * (self.rect.width() / pix_w)
                global_y = self.rect.top() + pm_y * (self.rect.height() / pix_h)
                self.section_xctr = max(0, min(int(global_x), fw - 1))
                self.section_yctr = max(0, min(int(global_y), fh - 1))
                self.lineEditSectionX.setText(str(self.section_xctr))
                self.lineEditSectionY.setText(str(self.section_yctr))
                self.mouse_x = event.pos().x()
                self.mouse_y = event.pos().y()
                self.update_raw_plots(self.cam_controller.frame)

        return mousepress

    def label_mousemove(self):
        def mousemove(event):
            if event.buttons() & QtCore.Qt.RightButton:
                if self.target_scale > 0:
                    delta_x = event.pos().x() - self.last_drag_pos.x()
                    delta_y = event.pos().y() - self.last_drag_pos.y()
                    if self.labelImage.width() > 0:
                        scale_x = self.rect.width() / self.labelImage.width()
                        scale_y = self.rect.height() / self.labelImage.height()
                        self.pan_x -= int(delta_x * scale_x)
                        self.pan_y -= int(delta_y * scale_y)
                        self.last_drag_pos = event.pos()

        return mousemove

    def label_mousewheel(self):
        def mousewheel(event):
            label_w = self.labelImage.width()
            label_h = self.labelImage.height()
            pix_w = self.pixmap.width()
            pix_h = self.pixmap.height()
            if pix_w == 0 or pix_h == 0: return
            if label_w / label_h > pix_w / pix_h:
                scaled_w = int(label_h * pix_w / pix_h);
                scaled_h = label_h
            else:
                scaled_w = label_w;
                scaled_h = int(label_w * pix_h / pix_w)
            off_x = (label_w - scaled_w) // 2;
            off_y = (label_h - scaled_h) // 2
            click_x = event.pos().x() - off_x;
            click_y = event.pos().y() - off_y
            if 0 <= click_x <= scaled_w and 0 <= click_y <= scaled_h:
                pm_x = click_x * (pix_w / scaled_w)
                pm_y = click_y * (pix_h / scaled_h)
                self.zoom_x = self.rect.left() + pm_x * (self.rect.width() / pix_w)
                self.zoom_y = self.rect.top() + pm_y * (self.rect.height() / pix_h)
                if event.angleDelta().y() > 0:
                    self.target_scale += 1
                else:
                    self.target_scale -= 1

        return mousewheel

    def toQImage(self, copy=False):
        qim = QtGui.QImage(self.cam_controller.frame.data, self.cam_controller.frame.shape[1],
                           self.cam_controller.frame.shape[0], self.cam_controller.frame.strides[0],
                           QtGui.QImage.Format_Indexed8).rgbSwapped()
        return qim.copy() if copy else qim

    def handle_logging_toggle(self):
        if self.checkBoxLogData.isChecked():
            self.current_log_session = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_start_time = time.time()
            print(f"Logging started: {self.current_log_session}")
        else:
            print("Logging stopped.")

    def set_log_window(self):
        try:
            val = int(self.lineEditLogWindow.text())
            if val > 0: self.avg_win_width = val
        except:
            pass

    def unit_change(self):
        if self.radioButtonUnitUm.isChecked():
            self.unit = getattr(self.cam_controller, 'pixel_size', 1.85)
        else:
            self.unit = 1.0

    def line_edit_multi_fits(self):
        try:
            self.num_fits = int(self.lineEditMultiFits.text())
        except ValueError:
            self.num_fits = 1

    def section_center(self):
        fw = getattr(self.cam_controller, 'framewidth', 1280)
        fh = getattr(self.cam_controller, 'frameheight', 1024)
        self.section_xctr = fw // 2
        self.section_yctr = fh // 2
        self.target_scale = 0;
        self.zoom_needed = False;
        self.pan_x = 0;
        self.pan_y = 0

    def section_center_line_edit(self):
        try:
            self.section_xctr = int(self.lineEditSectionX.text())
            self.section_yctr = int(self.lineEditSectionY.text())
        except:
            pass

    def set_exptime(self):
        self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
        self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))

    def set_average_frames(self):
        self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())

    def checkbox_auto_exposure(self):
        if self.checkBoxAutoExposure.isChecked():
            self.lineEditExposureTime.setEnabled(False)
        else:
            self.lineEditExposureTime.setEnabled(True)

    def set_background(self):
        self.cam_controller.set_background()

    def clear_background(self):
        self.cam_controller.clear_background()

    def file_save(self, path=None):
        if not path:
            base_dir = os.getcwd();
            save_dir = os.path.join(base_dir, "Saved_Images")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_name = f"Image_{timestamp}.bmp"
            default_path = os.path.join(save_dir, default_name)
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.mainwindow, 'Save Image', default_path,
                                                                "Images (*.bmp *.png *.jpg)")
        else:
            filename = path
        if filename:
            if self.cam_controller.frame is not None:
                if not any(filename.lower().endswith(ext) for ext in ['.bmp', '.png', '.jpg']): filename += '.bmp'
                scale_factor = 4;
                h, w = self.cam_controller.frame.shape
                sharp_image = cv2.resize(self.cam_controller.frame, (w * scale_factor, h * scale_factor),
                                         interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(filename, sharp_image)

    def update_temperature(self):
        pass

    def update_plot(self):
        pass

    def update_plot_withoutfit(self):
        pass


# import time
# import os
# import datetime
# import numpy as np
# import cv2
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import QPainter, QPen, QColor
# from PyQt5.QtCore import QRect, QPoint
# from pyqtgraph import PlotWidget, mkPen
#
# from FlirWindow import Ui_MainWindow
# from fitgauss import fitgauss2d_section, fitgauss2d_multiple
#
#
# # --- CUSTOM WIDGET: SMART VIEWER ---
# class SmartViewer(QtWidgets.QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setScaledContents(False)
#         self.setAlignment(QtCore.Qt.AlignCenter)
#         self.setMouseTracking(True)
#         # Prevent window resizing loop
#         self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
#
#         self._pixmap = None
#         self.crosshair_enabled = True
#         self.crosshair_pos = None
#
#         self.draw_rect = QRect()
#
#     def set_image(self, pixmap):
#         self._pixmap = pixmap
#         self.update()
#
#     def set_crosshair(self, enabled, pos=None):
#         self.crosshair_enabled = enabled
#         if pos:
#             self.crosshair_pos = pos
#         self.update()
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         bg_color = self.palette().color(QtGui.QPalette.Window)
#         painter.fillRect(self.rect(), bg_color)
#
#         if self._pixmap and not self._pixmap.isNull():
#             # Aspect Ratio Fit
#             w_widget = self.width()
#             h_widget = self.height()
#             w_img = self._pixmap.width()
#             h_img = self._pixmap.height()
#
#             if w_img > 0 and h_img > 0:
#                 scale = min(w_widget / w_img, h_widget / h_img)
#             else:
#                 scale = 1
#
#             draw_w = int(w_img * scale)
#             draw_h = int(h_img * scale)
#
#             off_x = (w_widget - draw_w) // 2
#             off_y = (h_widget - draw_h) // 2
#
#             self.draw_rect = QRect(off_x, off_y, draw_w, draw_h)
#
#             target_rect = self.draw_rect
#             source_rect = QRect(0, 0, w_img, h_img)
#
#             # FastTransformation = Sharp Pixels
#             painter.drawPixmap(target_rect, self._pixmap, source_rect)
#
#             # Draw Crosshair
#             if self.crosshair_enabled and self.crosshair_pos:
#                 pen = QPen(QtCore.Qt.green)
#                 pen.setWidth(1)
#                 pen.setCosmetic(True)
#                 painter.setPen(pen)
#
#                 pm_x, pm_y = self.crosshair_pos
#
#                 screen_x = off_x + (pm_x * scale)
#                 screen_y = off_y + (pm_y * scale)
#
#                 ix, iy = int(screen_x), int(screen_y)
#
#                 # Constrain lines to image area
#                 img_top = off_y
#                 img_bottom = off_y + draw_h
#                 img_left = off_x
#                 img_right = off_x + draw_w
#
#                 if (img_left <= ix <= img_right) and (img_top <= iy <= img_bottom):
#                     painter.drawLine(ix, img_top, ix, img_bottom)
#                     painter.drawLine(img_left, iy, img_right, iy)
#
#         painter.end()
#
#
# class Ui_CustomWindow(Ui_MainWindow):
#
#     def custom_init(self, mainwindow):
#         self.mainwindow = mainwindow
#
#         self.old_label = self.labelImage
#         self.labelImage = SmartViewer(self.centralwidget)
#         self.labelImage.setSizePolicy(self.old_label.sizePolicy())
#         self.gridLayoutImage.replaceWidget(self.old_label, self.labelImage)
#         self.old_label.deleteLater()
#
#         self.stime = time.time()
#         self.nframe = 0
#         self.lasttime = time.time()
#         self.section_xctr = 0
#         self.section_yctr = 0
#         self.section_xdata = []
#         self.section_ydata = []
#         self.section_xcoord = []
#         self.section_ycoord = []
#         self.unit = 1.0
#         self.iter = -1
#
#         # Zoom Variables
#         self.loclist = []
#         self.zoom_x = 0
#         self.zoom_y = 0
#         self.current_scale = 0
#         self.target_scale = 0
#         self.mouse_x = 0
#         self.mouse_y = 0
#
#         # Pan Variables
#         self.pan_x = 0
#         self.pan_y = 0
#         self.last_drag_pos = QtCore.QPoint()
#
#         # Logging
#         self.current_log_session = None
#         self.log_start_time = 0.0
#         self.avg_win_width = 50
#
#         # --- MEMORY BUFFER (RAM) ---
#         # Holds the last N values for Mean/StdDev calculation
#         self.data_buffer_x = []
#         self.data_buffer_y = []
#
#         self.cam_controller.log = self.plainTextEditLog
#         self.cam_controller.initialize()
#
#         self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
#         self.zoom_needed = False
#
#         self.update_timer = QtCore.QTimer()
#         self.update_timer.timeout.connect(self.update_movie)
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#         self.lineEditExposureTime.returnPressed.connect(self.set_exptime)
#
#         self.plotx = PlotWidget(self.centralwidget)
#         self.gridLayoutImage.addWidget(self.plotx, 1, 0, 1, 1)
#         self.ploty = PlotWidget(self.centralwidget)
#         self.gridLayoutImage.addWidget(self.ploty, 0, 1, 1, 1)
#         self.ploty.getPlotItem().getViewBox().invertY(True)
#
#         self.sectionx_line = self.plotx.plot([], [])
#         self.sectiony_line = self.ploty.plot([], [])
#         self.sectionx_fit = self.plotx.plot([], [], pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#         self.sectiony_fit = self.ploty.plot([], [], pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#
#         self.num_fits = 1
#         self.slices = 40
#
#         self.labelImage.mousePressEvent = self.label_mousepress()
#         self.labelImage.mouseMoveEvent = self.label_mousemove()
#         self.labelImage.wheelEvent = self.label_mousewheel()
#
#         self.pushButtonSectionCenter.clicked.connect(self.section_center)
#         self.lineEditAverageFrames.returnPressed.connect(self.set_average_frames)
#
#         self.radioButtonUnitUm.toggled.connect(self.unit_change)
#         self.radioButtonUnitPixel.toggled.connect(self.unit_change)
#         self.checkBoxAutoExposure.stateChanged.connect(self.checkbox_auto_exposure)
#         self.lineEditMultiFits.returnPressed.connect(self.line_edit_multi_fits)
#         self.pushButtonSetBg.clicked.connect(self.set_background)
#         self.pushButtonClearBg.clicked.connect(self.clear_background)
#
#         if hasattr(self, 'actionsave'): self.actionsave.triggered.connect(self.file_save)
#         if hasattr(self, 'actionsave_image'): self.actionsave_image.triggered.connect(self.file_save)
#         if hasattr(self, 'actionsave_image_2'): self.actionsave_image_2.triggered.connect(self.file_save)
#
#         self.checkBoxLogData.stateChanged.connect(self.handle_logging_toggle)
#         self.lineEditLogWindow.returnPressed.connect(self.set_log_window)
#
#         mainwindow.setWindowIcon(QtGui.QIcon('logo.png'))
#         self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
#
#         # Directories
#         self.start_date = str(datetime.datetime.now()).replace(':', '-')
#         self.start_time = time.time()
#
#         if not os.path.exists('log_t_x_y_wx_wy_h'): os.makedirs('log_t_x_y_wx_wy_h')
#
#         self.unit_change()
#         self.set_log_window()
#
#     def start_continue(self):
#         self.cam_controller.start_continue()
#         self.pushButtonContinue.setText("Stop")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.stop_continue)
#         self.update_timer.start(30)
#         self.iter = 0
#         self.stime = time.time()
#
#     def stop_continue(self):
#         self.update_timer.stop()
#         self.cam_controller.stop_continue()
#         self.pushButtonContinue.setText("Start")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#     def update_movie(self):
#         self.cam_controller.acquire_continue()
#         frame = self.cam_controller.frame
#
#         if frame is None or frame.size == 0:
#             return
#
#         # --- FIT LOGIC ---
#         if self.checkBoxFit.isChecked():
#             if not self.zoom_needed:
#                 p, ier = fitgauss2d_section(np.arange(0, frame.shape[1]),
#                                             np.arange(0, frame.shape[0]), frame)
#             else:
#                 p, ier = fitgauss2d_section(np.arange(0, frame.shape[1]),
#                                             np.arange(0, frame.shape[0]), frame)
#
#             # --- 1. UPDATE BUFFER (RAM) ---
#             # Always update the buffer so stats are live and smooth
#             self.data_buffer_x.append(p[0])
#             self.data_buffer_y.append(p[1])
#
#             # Maintain window size
#             while len(self.data_buffer_x) > self.avg_win_width:
#                 self.data_buffer_x.pop(0)
#                 self.data_buffer_y.pop(0)
#
#             # Calculate Stats from RAM (Fast)
#             x_cent_avg = np.mean(self.data_buffer_x)
#             y_cent_avg = np.mean(self.data_buffer_y)
#             x_cent_stdv = np.std(self.data_buffer_x) if len(self.data_buffer_x) > 1 else 0.0
#             y_cent_stdv = np.std(self.data_buffer_y) if len(self.data_buffer_y) > 1 else 0.0
#
#             # --- 2. LOGGING (DISK) ---
#             # If checked, we ALSO write to the file.
#             if self.checkBoxLogData.isChecked() and self.current_log_session:
#                 elapsed_time = time.time() - self.log_start_time
#
#                 # Write Text Data to File
#                 log_path = 'log_t_x_y_wx_wy_h/' + self.current_log_session + '.txt'
#                 with open(log_path, 'a') as f:
#                     f.write(f"{elapsed_time:.4f} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]}\n")
#
#                 # Note: Image auto-save removed as per previous request.
#                 # Only text is logged automatically now.
#
#             # --- 3. UPDATE DISPLAY ---
#             if self.lineEditxCenter.parent() is not None:
#                 self.lineEditxCenter.setText('%.2f' % (p[0] * self.unit))
#                 self.lineEditAVGxCenter.setText('%.2f' % (x_cent_avg * self.unit))
#                 self.lineEditDxCenter.setText('%.2f' % (x_cent_stdv * self.unit))
#                 self.lineEdityCenter.setText('%.2f' % (p[1] * self.unit))
#                 self.lineEditAVGyCenter.setText('%.2f' % (y_cent_avg * self.unit))
#                 self.lineEditDyCenter.setText('%.2f' % (y_cent_stdv * self.unit))
#                 self.lineEditxWaist.setText('%.1f' % (p[2] * 2 * self.unit))
#                 self.lineEdityWaist.setText('%.1f' % (p[3] * 2 * self.unit))
#                 self.lineEditHeight.setText('%.1f' % p[4])
#
#             self.update_fit_plots(p, frame)
#         else:
#             self.update_raw_plots(frame)
#             self.lineEditAVGxCenter.setText('N/A')
#             self.lineEditAVGyCenter.setText('N/A')
#
#         self.iter += 1
#
#         # --- DISPLAY IMAGE ---
#         self.pixmap = QtGui.QPixmap(self.toQImage())
#         self.zoom()
#
#         if self.section_xctr != 0 and self.section_yctr != 0 and self.checkBoxCrosshair.isChecked():
#             self.draw_crosshair()
#         else:
#             self.labelImage.set_crosshair(False)
#
#         self.labelImage.set_image(self.pixmap)
#
#         if self.checkBoxAutoExposure.isChecked():
#             self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#
#     # --- ZOOM & PAN ---
#     def zoom(self):
#         self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
#
#         if self.target_scale <= 0:
#             self.loclist = []
#             self.target_scale = 0
#             self.current_scale = 0
#             self.pan_x = 0
#             self.pan_y = 0
#         else:
#             if self.target_scale > self.current_scale:
#                 self.loclist.append([self.zoom_x, self.zoom_y])
#             elif self.target_scale < self.current_scale:
#                 if self.loclist: self.loclist.pop()
#
#             self.current_scale = 0
#             ZOOM_FACTOR = 0.8
#
#             for i in self.loclist:
#                 self.current_scale += 1
#                 imgsize = (self.pixmap.width(), self.pixmap.height())
#                 [ix, iy] = i
#
#                 ix_curr = float(ix) * self.pixmap.width() / float(self.cam_controller.framewidth)
#                 iy_curr = float(iy) * self.pixmap.height() / float(self.cam_controller.frameheight)
#
#                 new_w = int(imgsize[0] * ZOOM_FACTOR)
#                 new_h = int(imgsize[1] * ZOOM_FACTOR)
#
#                 xcorner = int(ix_curr - new_w / 2)
#                 ycorner = int(iy_curr - new_h / 2)
#
#                 xcorner = max(0, min(xcorner, imgsize[0] - new_w))
#                 ycorner = max(0, min(ycorner, imgsize[1] - new_h))
#
#                 rect = QRect(xcorner, ycorner, new_w, new_h)
#
#                 self.pixmap = self.pixmap.copy(rect)
#                 self.rect.setCoords(self.rect.x() + rect.x(), self.rect.y() + rect.y(),
#                                     self.rect.x() + rect.right(), self.rect.y() + rect.bottom())
#
#         if self.pan_x != 0 or self.pan_y != 0:
#             shifted_rect = self.rect.translated(self.pan_x, self.pan_y)
#             if shifted_rect.left() < 0: shifted_rect.moveLeft(0)
#             if shifted_rect.top() < 0: shifted_rect.moveTop(0)
#             if shifted_rect.right() > self.cam_controller.framewidth: shifted_rect.moveRight(
#                 self.cam_controller.framewidth)
#             if shifted_rect.bottom() > self.cam_controller.frameheight: shifted_rect.moveBottom(
#                 self.cam_controller.frameheight)
#
#             self.rect = shifted_rect
#             raw_qimg = self.toQImage(copy=False)
#             self.pixmap = QtGui.QPixmap.fromImage(raw_qimg).copy(self.rect)
#
#         self.plotx.setXRange(self.rect.left(), self.rect.right())
#         self.ploty.setYRange(self.rect.top(), self.rect.bottom())
#
#     def draw_crosshair(self):
#         local_x = self.section_xctr - self.rect.left()
#         local_y = self.section_yctr - self.rect.top()
#
#         if self.rect.width() > 0 and self.rect.height() > 0:
#             pm_x = local_x * (self.pixmap.width() / self.rect.width())
#             pm_y = local_y * (self.pixmap.height() / self.rect.height())
#             self.labelImage.set_crosshair(True, (pm_x, pm_y))
#         else:
#             self.labelImage.set_crosshair(False)
#
#     def label_mousepress(self):
#         def mousepress(event):
#             if event.button() == QtCore.Qt.RightButton:
#                 self.last_drag_pos = event.pos()
#                 return
#
#             if event.button() == QtCore.Qt.LeftButton:
#                 label_w = self.labelImage.width()
#                 label_h = self.labelImage.height()
#                 pix_w = self.pixmap.width()
#                 pix_h = self.pixmap.height()
#                 if pix_w == 0 or pix_h == 0: return
#
#                 scale = min(label_w / pix_w, label_h / pix_h)
#                 draw_w = int(pix_w * scale)
#                 draw_h = int(pix_h * scale)
#                 off_x = (label_w - draw_w) // 2
#                 off_y = (label_h - draw_h) // 2
#
#                 click_x = event.pos().x() - off_x
#                 click_y = event.pos().y() - off_y
#
#                 if click_x < 0 or click_x > draw_w or click_y < 0 or click_y > draw_h: return
#
#                 pm_x = click_x / scale
#                 pm_y = click_y / scale
#
#                 global_x = self.rect.left() + pm_x * (self.rect.width() / pix_w)
#                 global_y = self.rect.top() + pm_y * (self.rect.height() / pix_h)
#
#                 self.section_xctr = max(0, min(int(global_x), self.cam_controller.framewidth - 1))
#                 self.section_yctr = max(0, min(int(global_y), self.cam_controller.frameheight - 1))
#
#                 self.lineEditSectionX.setText(str(self.section_xctr))
#                 self.lineEditSectionY.setText(str(self.section_yctr))
#
#                 self.mouse_x = event.pos().x()
#                 self.mouse_y = event.pos().y()
#                 self.update_plot()
#
#         return mousepress
#
#     def label_mousemove(self):
#         def mousemove(event):
#             if event.buttons() & QtCore.Qt.RightButton:
#                 if self.target_scale > 0:
#                     delta_x = event.pos().x() - self.last_drag_pos.x()
#                     delta_y = event.pos().y() - self.last_drag_pos.y()
#
#                     if self.labelImage.width() > 0:
#                         scale_x = self.rect.width() / self.labelImage.width()
#                         scale_y = self.rect.height() / self.labelImage.height()
#                         self.pan_x -= int(delta_x * scale_x)
#                         self.pan_y -= int(delta_y * scale_y)
#                         self.last_drag_pos = event.pos()
#
#         return mousemove
#
#     def label_mousewheel(self):
#         def mousewheel(event):
#             # 1. Get the geometry of the Label vs Image
#             label_w = self.labelImage.width()
#             label_h = self.labelImage.height()
#             pix_w = self.pixmap.width()
#             pix_h = self.pixmap.height()
#
#             if pix_w == 0 or pix_h == 0: return
#
#             # 2. Determine the scale and offsets (Black Bars)
#             if label_w / label_h > pix_w / pix_h:
#                 scaled_w = int(label_h * pix_w / pix_h)
#                 scaled_h = label_h
#             else:
#                 scaled_w = label_w
#                 scaled_h = int(label_w * pix_h / pix_w)
#
#             off_x = (label_w - scaled_w) // 2
#             off_y = (label_h - scaled_h) // 2
#
#             # 3. Calculate Mouse Position relative to the Image
#             click_x = event.pos().x() - off_x
#             click_y = event.pos().y() - off_y
#
#             # Only zoom if mouse is over the actual image
#             if 0 <= click_x <= scaled_w and 0 <= click_y <= scaled_h:
#                 # Map screen click -> Current Pixmap Coordinates
#                 pm_x = click_x * (pix_w / scaled_w)
#                 pm_y = click_y * (pix_h / scaled_h)
#
#                 # 4. Convert to Global Sensor Coordinates
#                 # This accounts for the current zoom/crop (self.rect)
#                 self.zoom_x = self.rect.left() + pm_x * (self.rect.width() / pix_w)
#                 self.zoom_y = self.rect.top() + pm_y * (self.rect.height() / pix_h)
#
#                 if event.angleDelta().y() > 0:
#                     self.target_scale += 1
#                 else:
#                     self.target_scale -= 1
#
#         return mousewheel
#
#     def toQImage(self, copy=False):
#         qim = QtGui.QImage(self.cam_controller.frame.data, self.cam_controller.frame.shape[1],
#                            self.cam_controller.frame.shape[0], self.cam_controller.frame.strides[0],
#                            QtGui.QImage.Format_Indexed8).rgbSwapped()
#         return qim.copy() if copy else qim
#
#     def handle_logging_toggle(self):
#         if self.checkBoxLogData.isChecked():
#             self.current_log_session = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             self.log_start_time = time.time()
#             print(f"Logging started: {self.current_log_session}")
#         else:
#             print("Logging stopped.")
#
#     def set_log_window(self):
#         try:
#             val = int(self.lineEditLogWindow.text())
#             if val > 0: self.avg_win_width = val
#         except:
#             pass
#
#     def unit_change(self):
#         if self.radioButtonUnitUm.isChecked():
#             self.unit = getattr(self.cam_controller, 'pixel_size', 1.85)
#         else:
#             self.unit = 1.0
#
#     def update_raw_plots(self, frame):
#         self.section_xcoord = np.arange(0, frame.shape[1])
#         self.section_xdata = frame[self.section_yctr, :]
#         self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
#         self.sectionx_fit.setData([], [])
#         self.section_ycoord = np.arange(0, frame.shape[0])
#         self.section_ydata = frame[:, self.section_xctr]
#         self.sectiony_line.setData(self.section_ydata, self.section_ycoord)
#         self.sectiony_fit.setData([], [])
#
#     def update_fit_plots(self, p, frame):
#         self.update_raw_plots(frame)
#
#         def gauss(x, A, mu, sigma, offset):
#             return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset
#
#         fit_x = gauss(self.section_xcoord, p[4], p[0], p[2], p[5])
#         self.sectionx_fit.setData(self.section_xcoord, fit_x)
#         fit_y = gauss(self.section_ycoord, p[4], p[1], p[3], p[5])
#         self.sectiony_fit.setData(fit_y, self.section_ycoord)
#
#     def section_center(self):
#         self.section_xctr = self.cam_controller.framewidth // 2
#         self.section_yctr = self.cam_controller.frameheight // 2
#         self.target_scale = 0
#         self.zoom_needed = False
#         self.pan_x = 0
#         self.pan_y = 0
#
#     def section_center_line_edit(self):
#         try:
#             self.section_xctr = int(self.lineEditSectionX.text())
#             self.section_yctr = int(self.lineEditSectionY.text())
#         except:
#             pass
#
#     def set_exptime(self):
#         self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#
#     def set_average_frames(self):
#         self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
#
#     def checkbox_auto_exposure(self):
#         if self.checkBoxAutoExposure.isChecked():
#             self.lineEditExposureTime.setEnabled(False)
#         else:
#             self.lineEditExposureTime.setEnabled(True)
#
#     def line_edit_multi_fits(self):
#         try:
#             self.num_fits = int(self.lineEditMultiFits.text())
#         except ValueError:
#             self.num_fits = 1
#
#     def set_background(self):
#         self.cam_controller.set_background()
#
#     def clear_background(self):
#         self.cam_controller.clear_background()
#
#     def file_save(self, path=None):
#         if not path:
#             base_dir = os.getcwd()
#             save_dir = os.path.join(base_dir, "Saved Images")
#             if not os.path.exists(save_dir): os.makedirs(save_dir)
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             default_name = f"Image_{timestamp}.bmp"
#             default_path = os.path.join(save_dir, default_name)
#             filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.mainwindow, 'Save Image', default_path,
#                                                                 "Images (*.bmp *.png *.jpg)")
#         else:
#             filename = path
#
#         if filename:
#             if self.cam_controller.frame is not None:
#                 if not any(filename.lower().endswith(ext) for ext in ['.bmp', '.png', '.jpg']): filename += '.bmp'
#                 scale_factor = 4
#                 h, w = self.cam_controller.frame.shape
#                 sharp_image = cv2.resize(self.cam_controller.frame, (w * scale_factor, h * scale_factor),
#                                          interpolation=cv2.INTER_NEAREST)
#                 cv2.imwrite(filename, sharp_image)
#
#     def update_temperature(self):
#         pass
#
#     def update_plot(self):
#         pass
#
#     def update_plot_withoutfit(self):
#         pass


# import time
# import os
# import datetime
# import numpy as np
# import cv2
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import QPainter, QPen
# from PyQt5.QtCore import QRect, QPoint
# from pyqtgraph import PlotWidget, mkPen
#
# # Import your existing UI layout and Gaussian fitting logic
# from FlirWindow import Ui_MainWindow
# from fitgauss import fitgauss2d_section, fitgauss2d_multiple
#
#
# # ==================================================================================
# # CLASS: SmartViewer
# # Purpose: A custom QLabel subclass that handles:
# #   1. Displaying the camera image with CORRECT Aspect Ratio (no stretching).
# #   2. "Sharp" rendering (Nearest Neighbor) to avoid anti-aliasing blur.
# #   3. Digital Zooming centered on the mouse cursor or view center.
# #   4. Drawing a "Crosshair" overlay that is independent of the image resolution.
# #   5. Converting Screen Mouse Coordinates <-> Real Sensor Pixel Coordinates.
# # ==================================================================================
# class SmartViewer(QtWidgets.QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setScaledContents(False)  # We handle scaling manually in paintEvent
#         self.setAlignment(QtCore.Qt.AlignCenter)
#         self.setMouseTracking(False)  # Only track mouse when button is held (dragging)
#
#         self._pixmap = None
#         self.crosshair_enabled = True
#
#         # --- STATE VARIABLES ---
#         # 1. Crosshair Position: The specific pixel (x, y) on the sensor the crosshair points to.
#         self.sensor_cursor_x = 0
#         self.sensor_cursor_y = 0
#
#         # 2. View Position: The specific pixel (x, y) on the sensor that is in the CENTER of the screen.
#         self.view_x = 0
#         self.view_y = 0
#
#         # 3. Zoom Factor: 1.0 = Full Sensor Width fits in window. >1.0 = Zoomed in.
#         self.zoom_factor = 1.0
#         self.min_zoom = 1.0
#         self.max_zoom = 50.0
#
#         # --- DRAWING RECTANGLES (For coordinate mapping) ---
#         self._view_rect = QRect()  # Where on the screen the image is drawn (excludes gray bars)
#         self._source_rect = QRect()  # Which part of the sensor image is currently visible (ROI)
#
#     def set_image(self, pixmap):
#         """Receives the raw QPixmap from the camera loop."""
#         self._pixmap = pixmap
#         # If this is the very first frame, center the view and crosshair automatically
#         if self.view_x == 0 and self.view_y == 0 and pixmap:
#             cx = pixmap.width() // 2
#             cy = pixmap.height() // 2
#             self.view_x = cx
#             self.view_y = cy
#             self.sensor_cursor_x = cx
#             self.sensor_cursor_y = cy
#         self.update()  # Trigger paintEvent
#
#     def set_crosshair(self, enabled):
#         """Toggles the green crosshair visibility."""
#         self.crosshair_enabled = enabled
#         self.update()
#
#     def wheelEvent(self, event):
#         """Handles Mouse Wheel for Zooming."""
#         # 1. Determine which pixel is currently under the mouse cursor
#         mouse_sensor_pos = self.get_sensor_coords_from_mouse(event.pos())
#
#         old_zoom = self.zoom_factor
#
#         # 2. Update Zoom Factor based on scroll direction
#         delta = event.angleDelta().y()
#         if delta > 0:
#             self.zoom_factor *= 1.25
#         else:
#             self.zoom_factor /= 1.25
#
#         # Limit zoom range
#         self.zoom_factor = max(self.min_zoom, min(self.zoom_factor, self.max_zoom))
#
#         # 3. INTELLIGENT ZOOM CENTER:
#         # If the mouse is over the image, shift the view center (self.view_x/y)
#         # so that the pixel under the mouse STAYS under the mouse after zooming.
#         if mouse_sensor_pos is not None:
#             sx, sy = mouse_sensor_pos
#             ratio = old_zoom / self.zoom_factor
#             self.view_x += (sx - self.view_x) * (1 - ratio)
#             self.view_y += (sy - self.view_y) * (1 - ratio)
#
#         self.update()
#
#     def paintEvent(self, event):
#         """The core rendering loop. Draws background -> Image ROI -> Crosshair."""
#         painter = QPainter(self)
#
#         # 1. Fill Background with Theme Window Color (matches GUI gray)
#         bg_color = self.palette().color(QtGui.QPalette.Window)
#         painter.fillRect(self.rect(), bg_color)
#
#         if self._pixmap and not self._pixmap.isNull():
#             w_sensor = self._pixmap.width()
#             h_sensor = self._pixmap.height()
#
#             # --- A. CALCULATE VIEWPORT ROI ---
#             # Determine size of the crop based on zoom level
#             roi_w = w_sensor / self.zoom_factor
#             roi_h = h_sensor / self.zoom_factor
#
#             # Center the ROI around self.view_x/y
#             roi_x = self.view_x - (roi_w / 2)
#             roi_y = self.view_y - (roi_h / 2)
#
#             # Clamp ROI to ensure we don't look outside the sensor bounds
#             roi_x = max(0, min(roi_x, w_sensor - roi_w))
#             roi_y = max(0, min(roi_y, h_sensor - roi_h))
#
#             source_rect = QRect(int(roi_x), int(roi_y), int(roi_w), int(roi_h))
#             self._source_rect = source_rect
#
#             # --- B. CROP & SCALE ---
#             # 1. Crop the raw image to the ROI
#             cropped_pix = self._pixmap.copy(source_rect)
#
#             # 2. Scale the crop to fit the widget size, maintaining aspect ratio.
#             # CRITICAL: Use Qt.FastTransformation to prevent "smoothing" or "blurring"
#             scaled_pix = cropped_pix.scaled(
#                 self.size(),
#                 QtCore.Qt.KeepAspectRatio,
#                 QtCore.Qt.FastTransformation
#             )
#
#             # 3. Calculate offsets to center the image in the widget (handling gray bars)
#             x_off = (self.width() - scaled_pix.width()) // 2
#             y_off = (self.height() - scaled_pix.height()) // 2
#
#             self._view_rect = QRect(x_off, y_off, scaled_pix.width(), scaled_pix.height())
#
#             # Draw the final image
#             painter.drawPixmap(x_off, y_off, scaled_pix)
#
#             # --- C. DRAW CROSSHAIR ---
#             if self.crosshair_enabled:
#                 pen = QPen(QtCore.Qt.green)
#                 pen.setWidth(1)
#                 pen.setCosmetic(True)  # Ensures line is always 1px wide regardless of scale
#                 painter.setPen(pen)
#
#                 # Map the global sensor cursor position to the current screen ROI
#                 rel_x = self.sensor_cursor_x - source_rect.x()
#                 rel_y = self.sensor_cursor_y - source_rect.y()
#
#                 scale_x = scaled_pix.width() / source_rect.width()
#                 scale_y = scaled_pix.height() / source_rect.height()
#
#                 screen_x = x_off + (rel_x * scale_x)
#                 screen_y = y_off + (rel_y * scale_y)
#
#                 ix, iy = int(screen_x), int(screen_y)
#
#                 # Draw lines extending to the edges of the widget
#                 painter.drawLine(ix, 0, ix, self.height())
#                 painter.drawLine(0, iy, self.width(), iy)
#
#         painter.end()
#
#     # --- MOUSE INTERACTIONS ---
#     def mousePressEvent(self, event):
#         if event.button() == QtCore.Qt.LeftButton:
#             # Left Click: Move the Crosshair to this spot
#             self.update_cursor_from_mouse(event.pos())
#             super().mousePressEvent(event)  # Propagate event so plots can update
#         elif event.button() == QtCore.Qt.RightButton:
#             # Right Click: Pan the view (center the camera on this spot)
#             self.pan_view_to_mouse(event.pos())
#
#     def mouseMoveEvent(self, event):
#         # Dragging (Left Button held): Update crosshair position continuously
#         self.update_cursor_from_mouse(event.pos())
#         super().mouseMoveEvent(event)
#
#     def update_cursor_from_mouse(self, pos):
#         coords = self.get_sensor_coords_from_mouse(pos)
#         if coords:
#             self.sensor_cursor_x, self.sensor_cursor_y = coords
#             self.update()
#
#     def pan_view_to_mouse(self, pos):
#         coords = self.get_sensor_coords_from_mouse(pos)
#         if coords:
#             self.view_x, self.view_y = coords
#             self.update()
#
#     def get_sensor_coords_from_mouse(self, pos):
#         """
#         Converts Screen Pixel Coordinates (Mouse) -> Sensor Pixel Coordinates.
#         Accounts for: Gray bars (offsets), Zoom level, and Panning offset.
#         """
#         if self._view_rect.isEmpty(): return None
#
#         # 1. Remove gray bar offsets
#         rx = pos.x() - self._view_rect.x()
#         ry = pos.y() - self._view_rect.y()
#
#         # 2. Apply scaling ratio
#         ratio_x = self._source_rect.width() / self._view_rect.width()
#         ratio_y = self._source_rect.height() / self._view_rect.height()
#
#         # 3. Add the offset of the current ROI (panning)
#         sx = self._source_rect.x() + (rx * ratio_x)
#         sy = self._source_rect.y() + (ry * ratio_y)
#
#         # 4. Clamp to valid sensor bounds
#         if self._pixmap:
#             w = self._pixmap.width()
#             h = self._pixmap.height()
#             return max(0, min(int(sx), w - 1)), max(0, min(int(sy), h - 1))
#         return None
#
#     def get_current_sensor_coords(self):
#         return self.sensor_cursor_x, self.sensor_cursor_y
#
#
# # ==================================================================================
# # CLASS: Ui_CustomWindow
# # Purpose: The main logic controller for the GUI. Connects the "FlirCamWindow.py"
# #          layout to the "FlirCamController" hardware logic.
# # ==================================================================================
# class Ui_CustomWindow(Ui_MainWindow):
#
#     def custom_init(self, mainwindow):
#         self.mainwindow = mainwindow
#
#         # --- 1. INJECT SMART VIEWER ---
#         # Replace the placeholder label from UI file with our custom SmartViewer
#         self.old_label = self.labelImage
#         self.labelImage = SmartViewer(self.centralwidget)
#         self.labelImage.setSizePolicy(self.old_label.sizePolicy())
#         self.gridLayoutImage.replaceWidget(self.old_label, self.labelImage)
#         self.old_label.deleteLater()
#         # ---------------------------
#
#         # --- 2. INITIALIZE CAMERA ---
#         self.cam_controller.log = self.plainTextEditLog
#         self.cam_controller.initialize()
#
#         # --- 3. SETUP UPDATE LOOP ---
#         self.update_timer = QtCore.QTimer()
#         self.update_timer.timeout.connect(self.update_movie)  # Main loop function
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#         # --- 4. INIT UI CONTROLS ---
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#         self.lineEditExposureTime.returnPressed.connect(self.set_exptime)
#
#         # Setup PyQtGraph plots
#         self.plotx = PlotWidget(self.centralwidget)
#         self.gridLayoutImage.addWidget(self.plotx, 1, 0, 1, 1)
#         self.ploty = PlotWidget(self.centralwidget)
#         self.gridLayoutImage.addWidget(self.ploty, 0, 1, 1, 1)
#         self.ploty.getPlotItem().getViewBox().invertY(True)
#
#         # Initialize Plot Data Containers
#         self.section_xcoord = []
#         self.section_xdata = []
#         self.section_ycoord = []
#         self.section_ydata = []
#         self.sectionx_line = self.plotx.plot(self.section_xcoord, self.section_xdata)
#         self.sectiony_line = self.ploty.plot(self.section_ydata, self.section_ycoord)
#
#         self.sectionx_fit = self.plotx.plot([], [], pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#         self.sectiony_fit = self.ploty.plot([], [], pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#
#         self.section_xctr = 0
#         self.section_yctr = 0
#         self.num_fits = 1
#         self.unit = 1.0  # Default to pixels
#         self.slices = 40
#
#         # --- MEMORY BUFFERS (For Live Stats) ---
#         self.data_buffer_x = []
#         self.data_buffer_y = []
#
#         # --- 5. CONNECT SIGNALS & SLOTS ---
#         # Mouse interactions are handled via the SmartViewer wrapper
#         self.labelImage.mousePressEvent = self.handle_mouse_interaction
#         self.labelImage.mouseMoveEvent = self.handle_mouse_interaction
#
#         self.pushButtonSectionCenter.clicked.connect(self.section_center)
#         self.lineEditAverageFrames.returnPressed.connect(self.set_average_frames)
#
#         self.radioButtonUnitUm.toggled.connect(self.unit_change)
#         self.radioButtonUnitPixel.toggled.connect(self.unit_change)
#         self.checkBoxAutoExposure.stateChanged.connect(self.checkbox_auto_exposure)
#         self.lineEditMultiFits.returnPressed.connect(self.line_edit_multi_fits)
#         self.pushButtonSetBg.clicked.connect(self.set_background)
#         self.pushButtonClearBg.clicked.connect(self.clear_background)
#
#         # Save File Connections
#         if hasattr(self, 'actionsave'): self.actionsave.triggered.connect(self.file_save)
#         if hasattr(self, 'actionsave_image'): self.actionsave_image.triggered.connect(self.file_save)
#         if hasattr(self, 'actionsave_image_2'): self.actionsave_image_2.triggered.connect(self.file_save)
#
#         mainwindow.setWindowIcon(QtGui.QIcon('logo.png'))
#         self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
#
#         # Directories
#         self.start_date = str(datetime.datetime.now()).replace(':', '-')
#         self.start_time = time.time()
#
#         if not os.path.exists('log_t_x_y_wx_wy_h'): os.makedirs('log_t_x_y_wx_wy_h')
#         if not os.path.exists('logged_images_auto'): os.makedirs('logged_images_auto')
#
#         self.unit_change()
#         self.set_log_window() # Initialize window size from UI
#
#     def handle_logging_toggle(self):
#         if self.checkBoxLogData.isChecked():
#             # Create New Session
#             self.current_log_session = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             self.log_start_time = time.time()
#             print(f"Logging started: {self.current_log_session}")
#         else:
#             print("Logging stopped.")
#
#     def set_log_window(self):
#         try:
#             val = int(self.lineEditLogWindow.text())
#             if val > 0:
#                 self.avg_win_width = val
#                 # Trim live buffers immediately if necessary
#                 while len(self.data_buffer_x) > self.avg_win_width:
#                     self.data_buffer_x.pop(0)
#                     self.data_buffer_y.pop(0)
#             else:
#                 # Revert if invalid
#                 self.lineEditLogWindow.setText(str(self.avg_win_width))
#         except:
#             self.lineEditLogWindow.setText(str(self.avg_win_width))
#
#     def start_continue(self):
#         """Starts the continuous video acquisition loop."""
#         self.cam_controller.start_continue()
#         self.pushButtonContinue.setText("Stop")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.stop_continue)
#         self.update_timer.start(30)  # ~33 FPS
#
#     def stop_continue(self):
#         """Stops the video acquisition."""
#         self.update_timer.stop()
#         self.cam_controller.stop_continue()
#         self.pushButtonContinue.setText("Start")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#     def update_movie(self):
#         self.cam_controller.acquire_continue()
#         frame = self.cam_controller.frame
#
#         if frame is None or frame.size == 0:
#             return
#
#         # --- FIT LOGIC ---
#         if self.checkBoxFit.isChecked():
#             p, ier = fitgauss2d_section(np.arange(0, frame.shape[1]),
#                                         np.arange(0, frame.shape[0]), frame)
#
#             x_cent_avg, y_cent_avg = p[0], p[1]
#             x_cent_stdv, y_cent_stdv = 0.0, 0.0
#
#             # MODE 1: LOGGING ON (Read from File)
#             if self.checkBoxLogData.isChecked() and self.current_log_session:
#                 elapsed_time = time.time() - self.log_start_time
#
#                 # A. Write to File
#                 log_path = 'log_t_x_y_wx_wy_h/' + self.current_log_session + '.txt'
#                 with open(log_path, 'a') as f:
#                     f.write(f"{elapsed_time:.4f} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]}\n")
#
#                 # B. Save Image
#                 img_session_dir = os.path.join('logged_images_auto', self.current_log_session)
#                 if not os.path.exists(img_session_dir): os.makedirs(img_session_dir)
#                 img_name = f"img_{datetime.datetime.now().strftime('%H-%M-%S-%f')}.bmp"
#
#                 if not frame.flags['C_CONTIGUOUS']: frame_save = np.ascontiguousarray(frame)
#                 else: frame_save = frame
#                 cv2.imwrite(os.path.join(img_session_dir, img_name), frame_save)
#
#                 # C. Read Stats from File (Strict Requirement)
#                 try:
#                     with open(log_path, 'r') as f:
#                         lines = f.readlines()
#                         if len(lines) > self.avg_win_width:
#                             lines = lines[-self.avg_win_width:]
#
#                         floats_x = [float(l.split()[1]) for l in lines]
#                         floats_y = [float(l.split()[2]) for l in lines]
#
#                         x_cent_avg = np.mean(floats_x)
#                         y_cent_avg = np.mean(floats_y)
#                         x_cent_stdv = np.std(floats_x) if len(floats_x) > 1 else 0.0
#                         y_cent_stdv = np.std(floats_y) if len(floats_y) > 1 else 0.0
#                 except: pass
#
#             # MODE 2: LOGGING OFF (Live Preview Stats in RAM)
#             else:
#                 self.data_buffer_x.append(p[0])
#                 self.data_buffer_y.append(p[1])
#
#                 if len(self.data_buffer_x) > self.avg_win_width:
#                     self.data_buffer_x.pop(0)
#                     self.data_buffer_y.pop(0)
#
#                 x_cent_avg = np.mean(self.data_buffer_x) if self.data_buffer_x else p[0]
#                 y_cent_avg = np.mean(self.data_buffer_y) if self.data_buffer_y else p[1]
#                 x_cent_stdv = np.std(self.data_buffer_x) if len(self.data_buffer_x) > 1 else 0.0
#                 y_cent_stdv = np.std(self.data_buffer_y) if len(self.data_buffer_y) > 1 else 0.0
#
#             # Update GUI Text
#             if self.lineEditxCenter.parent() is not None:
#                 self.lineEditxCenter.setText('%.2f' % (p[0] * self.unit))
#                 self.lineEditAVGxCenter.setText('%.2f' % (x_cent_avg * self.unit))
#                 self.lineEditDxCenter.setText('%.2f' % (x_cent_stdv * self.unit))
#
#                 self.lineEdityCenter.setText('%.2f' % (p[1] * self.unit))
#                 self.lineEditAVGyCenter.setText('%.2f' % (y_cent_avg * self.unit))
#                 self.lineEditDyCenter.setText('%.2f' % (y_cent_stdv * self.unit))
#
#                 self.lineEditxWaist.setText('%.1f' % (p[2] * 2 * self.unit))
#                 self.lineEdityWaist.setText('%.1f' % (p[3] * 2 * self.unit))
#                 self.lineEditHeight.setText('%.1f' % p[4])
#
#             self.update_fit_plots(p, frame)
#         else:
#             self.update_raw_plots(frame)
#             self.lineEditAVGxCenter.setText('N/A')
#             self.lineEditAVGyCenter.setText('N/A')
#
#         # --- DISPLAY IMAGE ---
#         if not frame.flags['C_CONTIGUOUS']:
#             frame = np.ascontiguousarray(frame)
#
#         h, w = frame.shape
#         qim = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_Grayscale8)
#
#         self.labelImage.set_crosshair(self.checkBoxCrosshair.isChecked())
#         self.labelImage.set_image(QtGui.QPixmap.fromImage(qim))
#
#         if self.checkBoxAutoExposure.isChecked():
#             self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#
#     def handle_mouse_interaction(self, event):
#         """Called when SmartViewer detects a click or drag."""
#         # 1. Let SmartViewer handle internal logic (update crosshair visual)
#         SmartViewer.mousePressEvent(self.labelImage, event)
#
#         # 2. Retrieve the *actual* sensor coordinates of the crosshair
#         sx, sy = self.labelImage.get_current_sensor_coords()
#         self.section_xctr = sx
#         self.section_yctr = sy
#
#         # 3. Update the UI position indicators
#         self.lineEditSectionX.setText(str(sx))
#         self.lineEditSectionY.setText(str(sy))
#
#         # 4. Force a plot update immediately so graphs match the new crosshair position
#         if self.cam_controller.frame is not None:
#             self.update_raw_plots(self.cam_controller.frame)
#
#     def unit_change(self):
#         """Toggles between Pixels (1.0) and Micrometers (pixel_size)."""
#         if self.radioButtonUnitUm.isChecked():
#             self.unit = getattr(self.cam_controller, 'pixel_size', 1.85)
#         else:
#             self.unit = 1.0
#
#     def update_raw_plots(self, frame):
#         """Updates the side plots with raw intensity data slices."""
#         # X-Slice
#         self.section_xcoord = np.arange(0, frame.shape[1])
#         self.section_xdata = frame[self.section_yctr, :]
#         self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
#         self.sectionx_fit.setData([], [])  # Clear fit line
#
#         # Y-Slice
#         self.section_ycoord = np.arange(0, frame.shape[0])
#         self.section_ydata = frame[:, self.section_xctr]
#         self.sectiony_line.setData(self.section_ydata, self.section_ycoord)
#         self.sectiony_fit.setData([], [])  # Clear fit line
#
#     def update_fit_plots(self, p, frame):
#         """Updates plots with both raw data and the Gaussian fit curve."""
#         self.update_raw_plots(frame)
#
#         def gauss(x, A, mu, sigma, offset):
#             return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset
#
#         # Plot X Fit
#         fit_x = gauss(self.section_xcoord, p[4], p[0], p[2], p[5])
#         self.sectionx_fit.setData(self.section_xcoord, fit_x)
#
#         # Plot Y Fit
#         fit_y = gauss(self.section_ycoord, p[4], p[1], p[3], p[5])
#         self.sectiony_fit.setData(fit_y, self.section_ycoord)
#
#     def section_center(self):
#         """Resets the crosshair and zoom to the center of the sensor."""
#         self.section_xctr = self.cam_controller.framewidth // 2
#         self.section_yctr = self.cam_controller.frameheight // 2
#
#         self.labelImage.sensor_cursor_x = self.section_xctr
#         self.labelImage.sensor_cursor_y = self.section_yctr
#         self.labelImage.view_x = self.section_xctr
#         self.labelImage.view_y = self.section_yctr
#         self.labelImage.zoom_factor = 1.0
#         self.labelImage.update()
#
#     def section_center_line_edit(self):
#         """Moves crosshair based on text input coordinates."""
#         try:
#             self.section_xctr = int(self.lineEditSectionX.text())
#             self.section_yctr = int(self.lineEditSectionY.text())
#             self.labelImage.sensor_cursor_x = self.section_xctr
#             self.labelImage.sensor_cursor_y = self.section_yctr
#             self.labelImage.update()
#         except:
#             pass
#
#     def set_exptime(self):
#         self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#
#     def set_average_frames(self):
#         self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
#
#     def checkbox_auto_exposure(self):
#         if self.checkBoxAutoExposure.isChecked():
#             self.lineEditExposureTime.setEnabled(False)
#         else:
#             self.lineEditExposureTime.setEnabled(True)
#
#     def line_edit_multi_fits(self):
#         try:
#             self.num_fits = int(self.lineEditMultiFits.text())
#         except ValueError:
#             self.num_fits = 1
#
#     def set_background(self):
#         self.cam_controller.set_background()
#
#     def clear_background(self):
#         self.cam_controller.clear_background()
#
#     def file_save(self, path=None):
#         """
#         Saves the current frame.
#         If path is provided, saves directly.
#         If path is None, opens a File Dialog with a default timestamped name.
#         """
#         if not path:
#             base_dir = os.getcwd()
#             save_dir = os.path.join(base_dir, "Saved Images")
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             default_name = f"Image_{timestamp}.bmp"
#             default_path = os.path.join(save_dir, default_name)
#
#             filename, _ = QtWidgets.QFileDialog.getSaveFileName(
#                 self.mainwindow,
#                 'Save Image',
#                 default_path,
#                 "Images (*.bmp *.png *.jpg)"
#             )
#         else:
#             filename = path
#
#         if filename:
#             if self.cam_controller.frame is not None:
#                 # Ensure .bmp extension if none provided
#                 if not any(filename.lower().endswith(ext) for ext in ['.bmp', '.png', '.jpg']):
#                     filename += '.bmp'
#                 cv2.imwrite(filename, self.cam_controller.frame)
#
#     def update_temperature(self):
#         if hasattr(self.cam_controller, 'get_temperature'):
#             self.cam_controller.get_temperature()


# # original
# import datetime
# import time
#
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import QPainter, QPen
# from PyQt5.QtCore import QRect, QPoint
# from PyQt5.QtWidgets import QScrollBar, QScrollArea, QWidget, QApplication
# from pyqtgraph import PlotWidget
# from pyqtgraph import mkPen
# from FlirWindow import Ui_MainWindow
# from fitgauss import fitgauss2d_section, gauss1d, fitgauss2d_multiple
# import numpy as np
# import os
# import cv2
# import math
# import statistics
# import matplotlib.pyplot as plt
#
# class Ui_CustomWindow(Ui_MainWindow):
#
#     def custom_init(self,mainwindow):
#         self.mainwindow = mainwindow
#         self.stime = time.time()
#         self.nframe = 0
#         self.lasttime = time.time()
#         self.section_xctr = 0
#         self.section_yctr = 0
#         self.section_xctr0 = 0
#         self.section_yctr0 = 0
#         self.section_xdata = []
#         self.section_ydata = []
#         self.section_xcoord = []
#         self.section_ycoord = []
#         self.save_dir = os.getcwd()
#         self.unit = 0
#         self.iter = -1
#
#         self.loclist = []
#         self.zoom_x = 0
#         self.zoom_y = 0
#         self.current_scale = 0
#         self.target_scale = 0
#         self.zoom_scale_x = 0.0
#         self.zoom_scale_y = 0.0
#
#         # multiple fits line edit
#         self.num_fits = 1
#         self.lineEditMultiFits.setText('1')
#         self.line_edit_multi_fits()
#         # self.lineEditMultiFits.stateChanged.connect(self.line_edit_multi_fits)
#
#         self.slices = 40
#         # self.lineEditSlices.setText('40')
#         # self.line_edit_slices()
#
#         # Data for the fitted gaussian
#         self.section_xdata_fit = []
#         self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
#         self.section_ydata_fit = []
#         self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]
#
#
#         #logo
#         mainwindow.setWindowIcon(QtGui.QIcon('logo.png'))
#
#         # logging
#         self.cam_controller.log = self.plainTextEditLog
#
#         #init
#         self.cam_controller.initialize()
#
#         # init start and stop continue button
#         self.update_timer = QtCore.QTimer()
#         self.update_timer.timeout.connect(self.update_movie)
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#         # init exposure time line edit
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#         self.lineEditExposureTime.returnPressed.connect(self.set_exptime)
#
#         # init section plots
#         self.plotx = PlotWidget(self.centralwidget)
#         self.plotx.setObjectName("plotx")
#         self.gridLayoutImage.addWidget(self.plotx, 1, 0, 1, 1)
#
#         self.colors = [(np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(100, 256))
#                        for _ in range(self.num_fits)]
#
#         self.sectionx_fit = self.plotx.plot(self.section_xcoord, self.section_xdata_fit,
#                                             pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#         self.sectionx_fit_mult = [self.plotx.plot(self.section_xcoord, self.section_xdata_fit_mult[i],
#                                                   pen=mkPen(color=self.colors[i], style=QtCore.Qt.DashLine))
#                                   for i in range(len(self.section_xdata_fit_mult))]
#
#         self.sectionx_line = self.plotx.plot(self.section_xcoord,self.section_xdata)
#
#         self.ploty = PlotWidget(self.centralwidget)
#
#         self.ploty.setObjectName("ploty")
#         self.gridLayoutImage.addWidget(self.ploty,  0, 1, 1, 1)
#         self.ploty.getPlotItem().getViewBox().invertY(True)
#
#         self.sectiony_fit = self.ploty.plot(self.section_ydata_fit, self.section_ycoord,
#                                             pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#         self.sectiony_fit_mult = [self.ploty.plot(self.section_ydata_fit_mult[i], self.section_ycoord,
#                                                   pen=mkPen(color=self.colors[i], style=QtCore.Qt.DashLine))
#                                   for i in range(len(self.section_ydata_fit_mult))]
#
#         self.sectiony_line = self.ploty.plot(self.section_ydata,self.section_ycoord)
#
#         # Display fit parameters for multiple fits in colors that match the plots
#         self.param_displays = [[] for _ in range(5)]
#         self.param_labels = ('xCenter', 'yCenter', 'xWaist', 'yWaist', 'Height')
#         self.param_layout = QtWidgets.QGridLayout()
#         self.param_layout.setObjectName('gridLayoutParameters')
#         self.param_label_layout = QtWidgets.QGridLayout()
#         self.param_label_layout.setObjectName('gridLayoutParameterLabels')
#         # if True:
#         #     self.param_displays = [[QtWidgets.QLabel(self.centralwidget) for _ in range(self.num_fits)] for _ in range(5)]
#         #     for i in range(len(self.param_displays)):
#         #         for j in range(len(self.param_displays[i])):
#         #             self.param_displays[i][j].setFont(self.param_font)
#         #             self.param_displays[i][j].setObjectName("display" + self.param_labels[i] + str(j + 1))
#         #             self.param_displays[i][j].setStyleSheet('color: rgb(' + str(self.colors[j][0]) + ', '
#         #                                                     + str(self.colors[j][1]) + ', ' + str(self.colors[j][2]) + ');')
#         #             self.gridLayoutFitResult.addWidget(self.param_displays[i][j], i, j + 1, 1, 1)
#
#         # set section to the center
#         self.pushButtonSectionCenter.clicked.connect(self.section_center)
#
#         # set section center by line edit
#         self.lineEditSectionX.returnPressed.connect(self.section_center_line_edit)
#         self.lineEditSectionY.returnPressed.connect(self.section_center_line_edit)
#
#         # image label mouse press event
#         self.labelImage.mousePressEvent = self.label_mousepress()
#
#         # Image label mouse scroll event?
#         self.labelImage.wheelEvent = self.label_mousewheel()
#
#         # Zoom rectangle
#         self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
#         self.zoom_needed = False
#
#         # auto exposure check box
#         self.checkbox_auto_exposure()
#         self.checkBoxAutoExposure.stateChanged.connect(self.checkbox_auto_exposure)
#
#         # background
#         self.pushButtonSetBg.clicked.connect(self.set_background)
#         self.pushButtonClearBg.clicked.connect(self.clear_background)
#
#         # average frames
#         self.cam_controller.set_average_frames(self.lineEditAverageFrames.text())
#         self.lineEditAverageFrames.returnPressed.connect(self.set_average_frames)
#
#         # save file
#         self.actionsave_image_2.setStatusTip('Save File')
#         self.actionsave_image_2.triggered.connect(self.file_save)
#
#         # unit
#         self.unit_change()
#         self.radioButtonUnitPixel.toggled.connect(self.unit_change)
#
#         # temperature monitor
#         self.temperature_timer = QtCore.QTimer()
#         self.temperature_timer.timeout.connect(self.update_temperature)
#         self.temperature_timer.start(1000)
#
#         # center logging
#         self.start_date = str(datetime.datetime.now()).replace(':', '-')
#         self.start_time = time.time()
#         if not os.path.exists('center_log'):
#             os.makedirs('center_log')
#         with open('center_log/' + self.start_date + '.txt', 'w') as f:
#             f.write('time, x0, y0, x1, y1...\n')
#
#         # files containing only the x and y positions in the format:
#         if not os.path.exists('centers_x_log'):
#             os.makedirs('centers_x_log')
#         if not os.path.exists('centers_y_log'):
#             os.makedirs('centers_y_log')
#
#         # logging x_0 y_0 w_x w_y h
#         if not os.path.exists('log_x_y_wx_wy_h'):
#             os.makedirs('log_x_y_wx_wy_h')
#
#         self.avg_win_width = 50
#
#     def start_continue(self):
#         self.cam_controller.start_continue()
#         self.pushButtonContinue.setText("Stop Continue")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.stop_continue)
#         # self.t0 = time.time()
#         self.update_timer.start(10) # It delays the next update of the window by 10ms.
#         # Originally 500ms, that's the main reason why it has low frame rate.
#         # print('click')
#         self.iter = 0
#         self.stime = time.time()
#
#     def stop_continue(self):
#         self.update_timer.stop()
#         # print("--- %.8f seconds ---" % (time.time() - self.t0))
#         # print("no frames : %d"%(self.cam_controller.framecount))
#         self.cam_controller.stop_continue()
#         self.pushButtonContinue.setText("Start Continue")
#         self.pushButtonContinue.clicked.disconnect()
#         self.pushButtonContinue.clicked.connect(self.start_continue)
#
#     def update_movie(self):
#         # stime=time.time() # debugging
#
#         self.cam_controller.acquire_continue()
#         # print('1:' + str((time.time() - stime) * 1000))
#         if self.checkBoxFit.isChecked():
#             self.line_edit_multi_fits()
#             # self.line_edit_slices()
#             # print('2:' + str((time.time() - stime) * 1000))
#             if self.num_fits <= 1 or self.iter <= 0:
#                 print
#                 self.p, self.ier = fitgauss2d_section(np.arange(0, self.cam_controller.frame.shape[1]),
#                                             np.arange(0, self.cam_controller.frame.shape[0]), self.cam_controller.frame)
#                 # print('3:' + str((time.time() - stime) * 1000))
#                 if self.num_fits <= 1:
#                     self.update_plot()
#
#                     # add x and y centers to corresponding files in their own directories
#                     with open('centers_x_log/' + self.start_date + '.txt', 'a') as f:
#                         f.write(str(time.time() - self.start_time) + " " + str(self.p[0]) + "\n")
#                         f.flush() # save data immediately
#                     with open('centers_y_log/' + self.start_date + '.txt', 'a') as f:
#                         f.write(str(time.time() - self.start_time) + " " + str(self.p[1]) + "\n")
#                         f.flush()
#
#                     with open('log_x_y_wx_wy_h/' + self.start_date + '.txt', 'a') as f:
#                         f.write(str(time.time() - self.start_time) + " " + str(self.p[0]) + " " + str(self.p[1]) + " " + str(self.p[2]) + " " + str(self.p[3]) + " " + str(self.p[4]) + "\n")
#                         f.flush()
#
#                     # extract at most the last self.avg_win_width elements from the data files
#                     with open('centers_x_log/' + self.start_date + '.txt', 'r') as fx, open('centers_y_log/' + self.start_date + '.txt', 'r') as fy:
#                         i = 0
#                         line_x = []
#                         line_y = []
#
#                         for _ in fx: # only keep the last self.avg_win_width entries
#                             line_x = fx.readlines()
#                             line_y = fy.readlines()
#                             if len(line_x) > self.avg_win_width and len(line_y) > self.avg_win_width:
#                                 line_x = line_x[-self.avg_win_width:]
#                                 line_y = line_y[-self.avg_win_width:]
#
#                     # calculate the avg and stdev
#                     floats_x = [float(line.strip().split()[-1]) for line in line_x] # obtain only the values (last element in each row); convert strings to floats
#                     floats_y = [float(line.strip().split()[-1]) for line in line_y]
#                     self.x_cent_avg = np.mean(floats_x)
#                     self.y_cent_avg = np.mean(floats_y)
#                     self.x_cent_stdv = np.std(floats_x) if len(floats_x) > 1 else float("nan")
#                     self.y_cent_stdv = np.std(floats_y) if len(floats_y) > 1 else float("nan")
#
#                     # Display measurements in pixels instead of microns (if we want to convert back to microns, multiply by self.unit)
#                     if self.lineEditxCenter.parent() is not None:
#                         self.lineEditxCenter.setText('%.2f' % (self.p[0]))
#                         self.lineEditAVGxCenter.setText('%.2f' % (self.x_cent_avg))
#                         self.lineEditDxCenter.setText('%.2f' % (self.x_cent_stdv))
#                         self.lineEdityCenter.setText('%.2f' % (self.p[1]))
#                         self.lineEditAVGyCenter.setText('%.2f' % (self.y_cent_avg))
#                         self.lineEditDyCenter.setText('%.2f' % (self.y_cent_stdv))
#                         self.lineEditxWaist.setText('%.1f' % (self.p[2] * 2*self.unit))
#                         self.lineEdityWaist.setText('%.1f' % (self.p[3] * 2*self.unit))
#                         self.lineEditHeight.setText('%.1f' % (self.p[4]))
#                     else:
#                         self.update_param_displays()
#                 # print('4:' + str((time.time() - stime) * 1000))
#                 if self.iter <= 0:
#                     if self.p[2] < self.cam_controller.framewidth / self.slices and \
#                             self.p[3] < self.cam_controller.frameheight / self.slices:
#                         self.zoom_needed = True
#                         self.plainTextEditLog.insertPlainText(
#                             'Please zoom in to the features for multiple logging to work\n')
#                     else:
#                         self.zoom_needed = False
#             else:
#                 if not self.zoom_needed:
#                     self.p = fitgauss2d_multiple(self.cam_controller.frame, np.arange(self.cam_controller.frame.shape[0]),
#                                                  np.arange(self.cam_controller.frame.shape[1]), num_fits=self.num_fits,
#                                                  slices=self.slices)
#                 else:
#                     self.p = fitgauss2d_multiple(self.cam_controller.frame[self.rect.top():self.rect.bottom() + 1,
#                                                  self.rect.left():self.rect.right() + 1], np.arange(self.rect.height()),
#                                                  np.arange(self.rect.width()), num_fits=self.num_fits,
#                                                  slices=self.slices)
#                     for i in range(len(self.p)):
#                         self.p[i][0] += self.rect.left()
#                         self.p[i][1] += self.rect.top()
#
#                 self.update_plot()
#
#                 if self.lineEditxCenter.parent() is not None:
#                     self.gridLayoutFitResult.addLayout(self.param_label_layout, 0, 0, 1, 1)
#                     self.gridLayoutFitResult.addLayout(self.param_layout, 0, 1, 1, 1)
#                     self.lineEditxCenter.setParent(None)
#                     self.lineEditAVGxCenter.setParent(None)
#                     self.lineEditDxCenter.setParent(None)
#                     self.lineEdityCenter.setParent(None)
#                     self.lineEditAVGyCenter.setParent(None)
#                     self.lineEditDyCenter.setParent(None)
#                     self.lineEditxWaist.setParent(None)
#                     self.lineEdityWaist.setParent(None)
#                     self.lineEditHeight.setParent(None)
#
#                     self.labelAVG.setParent(None)
#                     self.param_label_layout.addWidget(self.labelAVG, 0, 2, 1, 1)
#                     self.labelD.setParent(None)
#                     self.param_label_layout.addWidget(self.labelD, 0, 3, 1, 1)
#                     self.labelxCenter.setParent(None)
#                     self.param_label_layout.addWidget(self.labelxCenter, 1, 0, 1, 1)
#                     self.labelyCenter.setParent(None)
#                     self.param_label_layout.addWidget(self.labelyCenter, 2, 0, 1, 1)
#                     self.labelxWaist.setParent(None)
#                     self.param_label_layout.addWidget(self.labelxWaist, 3, 0, 1, 1)
#                     self.labelyWaist.setParent(None)
#                     self.param_label_layout.addWidget(self.labelyWaist, 4, 0, 1, 1)
#                     self.labelHeight.setParent(None)
#                     self.param_label_layout.addWidget(self.labelHeight, 5, 0, 1, 1)
#
#                 self.update_param_displays()
#
#         else:
#             self.update_plot_withoutfit()
#             self.lineEditAVGxCenter.setText('N/A')
#             self.lineEditAVGyCenter.setText('N/A')
#             self.lineEditDxCenter.setText('N/A')
#             self.lineEditDyCenter.setText('N/A')
#             self.lineEditxCenter.setText('N/A')
#             self.lineEdityCenter.setText('N/A')
#             self.lineEditxWaist.setText('N/A')
#             self.lineEdityWaist.setText('N/A')
#             self.lineEditHeight.setText('N/A')
#
#         self.iter += 1
#
#         self.pixmap = QtGui.QPixmap(self.toQImage())
#         self.zoom()
#
#
#         #self.zoom_scale_x= float(self.pixmap.width())/float(self.cam_controller.framewidth)  #ratio of the displayed image to the original image
#         #self.zoom_scale_y=float(self.pixmap.height())/float(self.cam_controller.frameheight)  #inequality in zoom_scale_x and zoom_scale_y may indicate that the image is zoomed to very few pixels
#         #print(self.zoom_scale_x)
#         #print(self.zoom_scale_y)
#         if self.section_xctr != 0 and self.section_yctr != 0 and self.checkBoxCrosshair.isChecked():
#             self.draw_crosshair(self.mouse_x,self.mouse_y)
#         self.labelImage.setPixmap(self.pixmap)
#
#         # plt.plot(self.cam_controller.frame[round(p[1]),::])
#         # plt.plot(gauss1d(p[0],p[2],p[4],np.arange(0, self.cam_controller.frame.shape[1]))+p[5])
#         # plt.show()
#
#         if self.checkBoxAutoExposure.isChecked():
#             self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#         # self.nframe += 1
#         # print('5:' + str((time.time() - stime) * 1000))
#         # print(time.time()-self.lasttime)
#         # self.lasttime=time.time()
#         # print((time.time() - self.stime) * 1000 / self.nframe)
#
#     def update_param_displays(self, update_vals=True):
#         while self.num_fits > len(self.param_displays[0]):
#             for i in range(5):
#                 self.param_displays[i].append(QtWidgets.QLabel(self.centralwidget))
#                 self.param_displays[i][-1].setFont(self.param_font)
#                 self.param_displays[i][-1].setObjectName("display" + self.param_labels[i] +
#                                                          str(len(self.param_displays[i])))
#                 self.param_displays[i][-1].setStyleSheet('color: rgb(' +
#                                                          str(self.colors[len(self.param_displays[i]) - 1][0])
#                                                          + ', ' +
#                                                          str(self.colors[len(self.param_displays[i]) - 1][1])
#                                                          + ', ' +
#                                                          str(self.colors[len(self.param_displays[i]) - 1][2])
#                                                          + '); background-color: black')
#                 self.param_layout.addWidget(self.param_displays[i][-1], i, len(self.param_displays[i]),
#                                             1, 1)
#         while self.num_fits < len(self.param_displays[0]):
#             for i in range(5):
#                 self.param_displays[i][-1].setText('')
#                 self.param_displays[i].pop(-1)
#         for i in range(5):
#             for j in range(len(self.param_displays[i])):
#                 if update_vals:
#                     if 2 <= i <= 3:
#                         if self.num_fits > 1:
#                             self.param_displays[i][j].setText('%.1f' % (self.p[j][i] * 2 * self.unit))
#                         else:
#                             self.param_displays[i][j].setText('%.1f' % (self.p[i] * 2 * self.unit))
#                     else:
#                         if self.num_fits > 1:
#                             self.param_displays[i][j].setText('%.1f' % self.p[j][i])
#                         else:
#                             self.param_displays[i][j].setText('%.1f' % self.p[i])
#                 else:
#                     self.param_displays[i][j].setText('N/A')
#
#     def zoom(self):
#         self.rect = QRect(0, 0, self.cam_controller.framewidth, self.cam_controller.frameheight)
#         if self.target_scale <= 0:
#             self.loclist = []
#             self.xlist = []
#             self.ylist = []
#             self.target_scale = 0
#             self.current_scale = 0
#
#         else:
#             if self.target_scale > self.current_scale:
#                 self.loclist.append([self.zoom_x, self.zoom_y])
#             elif self.target_scale < self.current_scale:
#                 self.loclist.pop()
#             self.current_scale = 0
#             for i in self.loclist:
#                 self.current_scale += 1
#                 imgsize = (self.pixmap.width(), self.pixmap.height())
#                 [ix, iy] = i
#                 ix = float(ix) * self.pixmap.width() / float(self.cam_controller.framewidth)
#                 iy = float(iy) * self.pixmap.height() / float(self.cam_controller.frameheight)
#
#                 if ix < round(imgsize[0]/4):
#                     xcorner = 0
#                 elif ix > round(3*imgsize[0]/4):
#                     xcorner = round(imgsize[0]/2)
#                 else:
#                     xcorner = round(ix - imgsize[0]/4)
#                 if iy < round(imgsize[1]/4):
#                     ycorner = 0
#                 elif iy > round(3*imgsize[1]/4):
#                     ycorner = round(imgsize[1]/2)
#                 else:
#                     ycorner = round(iy - imgsize[1]/4)
#                 rect = QRect(xcorner,ycorner,2*imgsize[0]/4,2*imgsize[1]/4)
#
#                 #xoffset=min(ix,imgsize[0]-ix,imgsize[0]/4)                    #reduces the image to half or takes the boundary closest to the mouse point
#                 #yoffset = min(iy, imgsize[1] - iy, imgsize[1] / 4)             #change (1/4) to change the zoom step size
#                 #xoffset1=min(float(xoffset)/float(imgsize[0]),float(yoffset)/float(imgsize[1]))*imgsize[0]
#                 #yoffset1 = min(float(xoffset)/float(imgsize[0]), float(yoffset)/float(imgsize[1])) * imgsize[1]
#                 #rect = QRect(ix-xoffset1, iy-yoffset1, 2*xoffset1, 2*yoffset1)
#                 self.xlist.append(xcorner)
#                 self.ylist.append(ycorner)
#                 self.pixmap = self.pixmap.copy(rect)
#                 #self.xbox.append(self.pixmap.width())
#                 #self.ybox.append(self.pixmap.height())
#                 self.rect.setCoords(self.rect.x() + rect.x(), self.rect.y() + rect.y(), self.rect.x() + rect.right(),
#                                     self.rect.y() + rect.bottom())
#         self.plotx.setXRange(self.rect.left(), self.rect.right())
#         self.ploty.setYRange(self.rect.top(), self.rect.bottom())
#
#     def update_plot_withoutfit(self):
#         stime=time.time()
#         self.section_xcoord = np.arange(0, self.cam_controller.frame.shape[1])
#         self.section_xdata = self.cam_controller.frame[self.section_yctr,::]
#         self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
#         self.section_ycoord = np.arange(0, self.cam_controller.frame.shape[0])
#         self.section_ydata = self.cam_controller.frame[::,self.section_xctr]
#         self.sectiony_line.setData(self.section_ydata,self.section_ycoord)
#
#         self.section_xdata_fit = []
#         self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
#         self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
#         for i in range(len(self.sectionx_fit_mult)):
#             self.sectionx_fit_mult[i].setData(self.section_xcoord, [])
#         self.section_ydata_fit = []
#         self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
#         self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]
#         for i in range(len(self.sectiony_fit_mult)):
#             self.sectiony_fit_mult[i].setData(self.section_ydata_fit_mult[i], self.section_ycoord)
#         print('6:' + str((time.time() - stime) * 1000))
#
#     def update_plot(self):
#         # self.line_edit_multi_fits()
#         # self.line_edit_slices()
#         # stime=time.time()
#         self.section_xcoord = np.arange(0, self.cam_controller.frame.shape[1])
#         self.section_xdata = self.cam_controller.frame[self.section_yctr,::]
#         self.sectionx_line.setData(self.section_xcoord, self.section_xdata)
#         self.section_ycoord = np.arange(0, self.cam_controller.frame.shape[0])
#         self.section_ydata = self.cam_controller.frame[::,self.section_xctr]
#         self.sectiony_line.setData(self.section_ydata,self.section_ycoord)
#
#         # print('test2')
#
#         if self.num_fits <= 1 or not isinstance(self.p[0], list):
#             # Create the lines for the fitted graph to be plotted below the data
#             self.section_xdata_fit = self.gauss_data(self.p[4], self.p[0], self.p[2], self.p[5], self.section_xcoord.shape[0])
#             self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
#             self.section_xdata_fit_mult = [[] for _ in range(self.num_fits)]
#             for i in range(len(self.sectionx_fit_mult)):
#                 self.sectionx_fit_mult[i].setData([], [])
#
#             self.section_ydata_fit = self.gauss_data(self.p[4], self.p[1], self.p[3], self.p[5], self.section_ycoord.shape[0])
#             self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
#             self.section_ydata_fit_mult = [[] for _ in range(self.num_fits)]
#             for i in range(len(self.sectiony_fit_mult)):
#                 self.sectiony_fit_mult[i].setData([], [])
#         else:
#             self.section_xdata_fit = []
#             self.sectionx_fit.setData([], [])
#             self.section_ydata_fit = []
#             self.sectiony_fit.setData([], [])
#             self.section_xdata_fit_mult = [[] for _ in range(len(self.p))]
#             for i in range(len(self.sectionx_fit_mult)):
#                 self.sectionx_fit_mult[i].setData([], [])
#             self.section_ydata_fit_mult = [[] for _ in range(len(self.p))]
#             for i in range(len(self.sectiony_fit_mult)):
#                 self.sectiony_fit_mult[i].setData([], [])
#             for fit_num in range(len(self.p)):
#                 # print('test3')
#                 # self.section_xdata_fit = (self.gauss_data(fit[4], fit[0], fit[2], fit[5], self.section_xcoord.shape[0]))
#                 # self.sectionx_fit.setData(self.section_xcoord, self.section_xdata_fit)
#                 # self.plotx.plot(self.section_xcoord, self.section_xdata_fit,
#                 #                 pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#                 self.section_xdata_fit_mult[fit_num] = self.gauss_data(self.p[fit_num][4], self.p[fit_num][0],
#                                                                        self.p[fit_num][2], self.p[fit_num][5],
#                                                                        self.section_xcoord.shape[0])
#                 if len(self.colors) <= fit_num:
#                     self.colors.append((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
#                 if len(self.sectionx_fit_mult) <= fit_num:
#                     self.sectionx_fit_mult.append(self.plotx.plot(self.section_xcoord,
#                                                                   self.section_xdata_fit_mult[fit_num],
#                                                   pen=mkPen(color=self.colors[-1], style=QtCore.Qt.DashLine)))
#                 else:
#                     self.sectionx_fit_mult[fit_num].setData(self.section_xcoord, self.section_xdata_fit_mult[fit_num])
#
#                 # self.section_ydata_fit = (self.gauss_data(fit[4], fit[1], fit[3], fit[5], self.section_ycoord.shape[0]))
#                 # self.sectiony_fit.setData(self.section_ydata_fit, self.section_ycoord)
#                 # self.ploty.plot(self.section_ydata_fit, self.section_ycoord,
#                 #                 pen=mkPen(color=(0, 255, 0), style=QtCore.Qt.DashLine))
#                 self.section_ydata_fit_mult[fit_num] = self.gauss_data(self.p[fit_num][4], self.p[fit_num][1],
#                                                                        self.p[fit_num][3], self.p[fit_num][5],
#                                                                        self.section_ycoord.shape[0])
#                 if len(self.sectiony_fit_mult) <= fit_num:
#                     self.sectiony_fit_mult.append(self.ploty.plot(self.section_ydata_fit_mult[fit_num],
#                                                                   self.section_ycoord,
#                                                                   pen=mkPen(color=self.colors[-1],
#                                                                             style=QtCore.Qt.DashLine)))
#                 else:
#                     self.sectiony_fit_mult[fit_num].setData(self.section_ydata_fit_mult[fit_num], self.section_ycoord)
#
#         # center logging
#         with open('center_log/' + self.start_date + '.txt', 'a') as f:
#             if self.num_fits <= 1 or not isinstance(self.p[0], list):
#                 f.write(str(time.time() - self.start_time) + ', ' + str(self.p[0]) + ', ' + str(self.p[1]) + '\n')
#             else:
#                 f.write(str(time.time() - self.start_time))
#                 for fit in self.p:
#                     f.write(', ' + str(fit[0]) + ', ' + str(fit[1]))
#                 f.write('\n')
#         # print('7:' + str((time.time() - stime) * 1000))
#
#     # Gaussian function
#     def gauss(self, t, A, mu, sigma, offset):
#         return A * (math.e ** (-1 * ((t - mu) ** 2) / (2 * (sigma ** 2)))) + offset
#
#     # Plot a gaussian over a range [0, data_range)
#     def gauss_data(self, A, mu, sigma, offset, data_range):
#         out = []
#         for i in range(data_range):
#             out.append(self.gauss(i, A, mu, sigma, offset))
#         return out
#
#     def toQImage(self, copy=False):
#         '''
#         Transfer the format of the frame from numpy.ndarray to QImage
#         :param self:
#         :param copy:
#         :return:
#         '''
#
#         qim = QtGui.QImage(self.cam_controller.frame.data, self.cam_controller.frame.shape[1],
#                            self.cam_controller.frame.shape[0], self.cam_controller.frame.strides[0],
#                            QtGui.QImage.Format_Indexed8).rgbSwapped()
#
#
#         # qim.setColorTable(gray_color_table)
#         return qim.copy() if copy else qim
#
#     def draw_crosshair(self,mouse_x,mouse_y):
#         label_width = self.labelImage.size().width()
#         label_height = self.labelImage.size().height()
#         painter = QtGui.QPainter(self.pixmap)
#         pen = QtGui.QPen()
#         pen.setWidth(1) #reduced width of crosshair from 20 to 5
#         pen.setColor(QtGui.QColor('green'))
#         painter.setPen(pen)
#         painter.setOpacity(0.4)
#         painter.drawLine((mouse_x*self.pixmap.width())/label_width,0,(mouse_x*self.pixmap.width())/label_width,self.pixmap.height())
#         painter.drawLine(0,(mouse_y*self.pixmap.height())/label_height,self.pixmap.width(),(mouse_y*self.pixmap.height())/label_height)
#         painter.end()
#
#     def set_exptime(self):
#         self.cam_controller.configure_exposure(self.lineEditExposureTime.text())
#         self.lineEditExposureTime.setText(str(self.cam_controller.get_exposure()))
#
#     def label_mousepress(self):
#
#         def mousepress(eventQMouseEvent):
#             label_width = self.labelImage.size().width()
#             label_height = self.labelImage.size().height()
#             self.mouse_x = eventQMouseEvent.pos().x()
#             self.mouse_y = eventQMouseEvent.pos().y()
#             if len(self.loclist) == 0:
#                 self.section_xctr = round(self.cam_controller.framewidth * self.mouse_x / label_width)
#                 self.section_yctr = round(self.cam_controller.frameheight * self.mouse_y / label_height)
#             else:
#                 self.section_xctr = round(self.pixmap.width()* self.mouse_x / label_width)
#                 self.section_yctr = round(self.pixmap.height()*self.mouse_y/label_height)
#                 for i in range(len(self.loclist)):
#                     self.section_xctr += self.xlist[len(self.xlist)-i-1]
#                     self.section_yctr += self.ylist[len(self.ylist)-i-1]
#             self.lineEditSectionX.setText(str(self.section_xctr))
#             self.lineEditSectionY.setText(str(self.section_yctr))
#
#
#         # def mousepress(eventQMouseEvent):
#         #     label_width = self.labelImage.size().width()
#         #     label_height = self.labelImage.size().height()
#         #     self.mouse_x = eventQMouseEvent.pos().x()
#         #     self.mouse_y = eventQMouseEvent.pos().y()
#         #     self.section_xctr = round(self.cam_controller.framewidth * self.mouse_x / label_width)
#         #     self.section_yctr = round(self.cam_controller.frameheight*self.mouse_y/label_height)
#         #
#         #     if len(self.loclist) == 0:
#         #         self.lineEditSectionX.setText(str(self.section_xctr))
#         #         self.lineEditSectionY.setText(str(self.section_yctr))
#         #     else:
#         #         self.section_xctr_pseudo = round(self.pixmap.width()* self.mouse_x / label_width)
#         #         self.section_yctr_pseudo = round(self.pixmap.height()*self.mouse_y/label_height)
#         #         for i in range(len(self.loclist)):
#         #                 self.section_xctr_pseudo += self.xlist[len(self.xlist)-i-1]
#         #                 self.section_yctr_pseudo += self.ylist[len(self.ylist)-i-1]
#         #         self.lineEditSectionX.setText(str(self.section_xctr_pseudo))
#         #         self.lineEditSectionY.setText(str(self.section_yctr_pseudo))
#
#
#             # if len(self.loclist) == 0:
#             #     self.section_xctr = round(self.cam_controller.framewidth*self.mouse_x/label_width)
#             #     self.section_yctr = round(self.cam_controller.frameheight*self.mouse_y/label_height)
#             # else:
#             #     self.section_xctr = round(self.pixmap.width()*self.mouse_x/label_width)
#             #     self.section_yctr = round(self.pixmap.height()*self.mouse_y/label_height)
#             #     for i in range(len(self.loclist)):
#             #         self.section_xctr += self.xlist[i]
#             #         self.section_yctr += self.ylist[i]
#             #         print(str(self.xlist[i]))
#             #         print(str(self.ylist[i]))
#             # self.lineEditSectionX.setText(str(self.section_xctr))
#             # self.lineEditSectionY.setText(str(self.section_yctr))
#
#
#             if self.checkBoxFit.isChecked():
#                 self.update_plot()
#             else:
#                 self.update_plot_withoutfit()
#         return mousepress
#
#     def label_mousewheel(self):
#         def mousewheel(event):
#             if self.checkBoxZoom.isChecked():
#                 screenpoint = self.labelImage.mapFromGlobal(QtGui.QCursor.pos())
#                 self.zoom_x, self.zoom_y = event.pos().x() + screenpoint.x(), event.pos().y() + screenpoint.y()
#                 self.zoom_x = float(self.zoom_x)*self.cam_controller.framewidth/(2*self.labelImage.size().width())  # scaling from mouse position to display position
#                 self.zoom_y = float(self.zoom_y)*self.cam_controller.frameheight/(2*self.labelImage.size().height())
#                 if event.angleDelta().y() > 0:
#                     if self.target_scale < 5:
#                         self.target_scale += 1
#                     else:
#                         print('Reached Zoom Limit!')
#                 else:
#                     self.target_scale -= 1
#                 self.update_movie()
#             else:
#                 self.update_movie()
#         return mousewheel
#
#     def section_center(self):
#         # self.section_xctr = round(float(self.lineEditxCenter.text())/self.unit)
#         # self.section_yctr = round(float(self.lineEdityCenter.text())/self.unit)
#
#         # Instead change center to the centers of each gaussian
#         if self.num_fits <= 1:
#             self.section_xctr = round(self.p[0])
#             self.section_yctr = round(self.p[1])
#         else:
#             self.section_xctr = round(self.p[0][0])
#             self.section_yctr = round(self.p[0][1])
#
#         self.lineEditSectionX.setText(str(self.section_xctr))
#         self.lineEditSectionY.setText(str(self.section_yctr))
#
#     def section_center_line_edit(self):
#         try:
#             self.section_xctr = round(float(self.lineEditSectionX.text())/self.unit)
#             self.section_yctr = round(float(self.lineEditSectionY.text())/self.unit)
#         except ValueError as ex:
#             self.lineEditSectionX(str(self.section_xctr))
#             self.lineEditSectionY(str(self.section_yctr))
#             self.plainTextEditLog.insertPlainText('ValueError: %s' %(ex))
#
#     def checkbox_auto_exposure(self):
#         if self.checkBoxAutoExposure.isChecked():
#             self.cam_controller.reset_exposure()
#         else:
#             self.set_exptime()
#
#     def line_edit_multi_fits(self):
#         try:
#             if int(self.lineEditMultiFits.text()) > 0:
#                 self.num_fits = int(self.lineEditMultiFits.text())
#             else:
#                 self.num_fits = 1
#         except ValueError:
#             self.plainTextEditLog.insertPlainText('Please enter an integer value for the number of fits\n')
#
#     # def line_edit_slices(self):
#     #     try:
#     #         if int(self.lineEditSlices.text()) > 0:
#     #             self.slices = int(self.lineEditSlices.text())
#     #         else:
#     #             self.slices = 40
#     #     except ValueError:
#     #         self.plainTextEditLog.insertPlainText('Please enter an integer value for the number of slices\n')
#
#     def set_background(self):
#         self.cam_controller.set_background()
#
#     def clear_background(self):
#         self.cam_controller.clear_background()
#
#     def set_average_frames(self):
#         if not self.cam_controller.set_average_frames(self.lineEditAverageFrames.text()):
#             self.lineEditAverageFrames.setText(str(self.cam_controller.average_frames))
#
#     def file_save(self):
#         frametosave = self.cam_controller.frame
#         filename = "cx"+self.lineEditxCenter.text()+"cy"+self.lineEdityCenter.text() \
#             + "wx" + self.lineEditxWaist.text() + "wy" +self.lineEdityWaist.text() \
#             + "h" + self.lineEditHeight.text()
#         if self.radioButtonUnitPixel.isChecked():
#             filename += "pixel"
#         else:
#             filename += "um"
#         filename  += ".jpg"
#         name = QtWidgets.QFileDialog.getSaveFileName(self.mainwindow, 'Save File', os.path.join(self.save_dir, filename),
#                                                  "Images (*.png *.jpg)")[0]
#         # name = QtGui.QFileDialog.getSaveFileName(self.mainwindow, 'Save File',os.path.join(self.save_dir,filename),"Images (*.png *.jpg)")[0]
#         if not name is '':
#             self.save_dir = os.path.split(name)[0]
#             cv2.imwrite(name,frametosave)
#             self.plainTextEditLog.insertPlainText('Picture saved to %s\n'%(name))
#
#     def unit_change(self):
#         if self.radioButtonUnitPixel.isChecked():
#             self.unit = 1
#         else:
#             self.unit = self.cam_controller.pixel_size
#
#     def update_temperature(self):
#         self.cam_controller.get_temperature()
#         if self.cam_controller.device_temperature > self.cam_controller.device_temp_lim:
#             self.statusbar.showMessage("Warning: device temperature too high! T = %.1f > %.1f"%(self.cam_controller.device_temperature,self.cam_controller.device_temp_lim))
#
