import PySpin
from fitgauss import gauss2d
from numpy import zeros, uint8
import numpy as np

class FakeCam:
    def ExposureTime(self):
        return 10


class FakeCamContoller:
    def initialize(self):
        # Flag representing the status of the continue mode
        self.flag_continue = False
        self.framewidth = 0
        self.frameheight = 0
        self.frame = []
        self.background = []
        self.nobackground = []
        self.framecount = 0
        self.exposuretimeupperlimit = 1000000
        self.average_frames = 0
        self.pixel_size = 1.85 #um
        self.device_temperature = 0
        self.device_temp_lim = 50

        # set frame size after binning
        self.framewidth = 2000
        self.frameheight = 1500
        self.frame = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.background = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.nobackground = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.floatzeroframe = zeros((self.frameheight, self.framewidth))

    def close(self):
        self.update_log('Camera closed...')

    def start_continue(self):
        # :param cam: Camera to run on.
        # :type cam: CameraPtr
        # :return: True if successful, False otherwise.
        # :rtype: bool
        if self.flag_continue:
            self.update_log('Acquisition already started...')
            return

        self.flag_continue = True
        self.update_log('Acquiring images...')

        return True

    def stop_continue(self):
        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        if self.flag_continue:
            self.flag_continue = False
            self.update_log('Stop acquiring images...')
        else:
            self.update_log('Acquiring already stopped...')
        return True

    def acquire_continue(self):
        xx, yy = np.meshgrid(np.arange(self.framewidth), np.arange(self.frameheight))
        image_data = (gauss2d(500, 800, 100, 400, 1, 0, xx, yy) + 1 + np.random.rand(*(xx.shape)) * 0.5)*50
        if self.average_frames > 1:
            image_data = image_data/self.average_frames
            for i in range(self.average_frames - 1):
                image_data += ((gauss2d(500, 800, 100, 400, 1, 0, xx, yy) + 1 + np.random.rand(*(xx.shape)) * 0.5)*50) / self.average_frames

        image_data = image_data.astype(np.uint8)
        self.frame = image_data
        self.framecount += 1

        return image_data

    def set_average_frames(self, average_frames_str):
        try:
            self.average_frames = max(1, int(average_frames_str))
        except ValueError as ex:
            self.update_log('ValueError: %s' % ex)
            return False
        self.update_log('Average frame number : %d'%(self.average_frames))
        return True

    def configure_exposure(self,exposure_time_to_set):
        try:
            result = True
            self.update_log('Exposure time set to %s us...\n' % 10)

        except PySpin.SpinnakerException as ex:
            self.update_log('Error: %s' % ex)
            result = False

        return result

    def reset_exposure(self):
        self.update_log('Automatic exposure enabled...')

        return True

    def get_exposure(self):
        return 10

    def get_temperature(self):
        self.device_temperature = 51
        return

    def set_background(self):
        self.background = self.frame

    def clear_background(self):
        self.background = self.nobackground
        
    def update_log(self, log):
        self.log.insertPlainText(log)
        self.log.insertPlainText('\n')
        
    def check_available_writable(self, node):
        return True

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    flir = FakeCamContoller()
    flir.start_continue()
    flir.acquire_continue()
    flir.acquire_continue()
    flir.acquire_continue()
    flir.update_log(flir.frame.shape)
    plt.imshow(flir.frame)
    plt.show()
    flir.stop_continue()
    flir.close()