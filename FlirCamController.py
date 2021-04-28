import PySpin
from numpy import zeros, uint8

class FlirCamController:
    def initialize(self):
        self.flag_continue = False  # Flag representing the status of the continue mode
        self.framewidth = 0
        self.frameheight = 0
        self.frame = []
        self.background = []
        self.nobackground = []
        self.floatzeroframe = []
        self.framecount = 0
        self.exposuretimeupperlimit = 1000000
        self.average_frames = 0
        self.pixel_size = 1.85  # um
        self.device_temperature = 0
        self.device_temp_lim = 50

        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        self.update_log('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()

        self.update_log('Number of cameras detected: %d' % num_cameras)

        if num_cameras == 0:
            # Finish if there are no cameras
            self.close()
            self.update_log('Not enough cameras!')
        else:
            # Choose the first camera
            self.cam = self.cam_list[0]

            # Initialize camera
            self.cam.Init()

        nodemap = self.cam.GetNodeMap()

        # Change binning mode to 2x2 to increase frame rate
        node_binninghorizontal = PySpin.CIntegerPtr(nodemap.GetNode('BinningHorizontal'))
        if not self.check_available_writable(node_binninghorizontal): return False
        node_binninghorizontal.SetValue(2)
        self.update_log(
            '%s is set to %f' % (node_binninghorizontal.GetDisplayName(), node_binninghorizontal.GetValue()))

        node_binningvertical = PySpin.CIntegerPtr(nodemap.GetNode('BinningVertical'))
        if not self.check_available_writable(node_binningvertical): return False
        node_binningvertical.SetValue(2)
        self.update_log('%s is set to %f' % (node_binningvertical.GetDisplayName(), node_binningvertical.GetValue()))

        # set frame size after binning
        self.framewidth = 2000
        self.frameheight = 1500
        self.pixel_size *= 2
        self.frame = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.background = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.nobackground = zeros((self.frameheight, self.framewidth), dtype=uint8)
        self.floatzeroframe = zeros((self.frameheight, self.framewidth))

        # set lower and upper limit of auto exposure time
        node_exposuretimelowerlimit = PySpin.CFloatPtr(nodemap.GetNode('AutoExposureExposureTimeLowerLimit'))
        if not self.check_available_writable(node_exposuretimelowerlimit): return False
        node_exposuretimelowerlimit.SetValue(self.cam.ExposureTime.GetMin())
        self.update_log(
            '%s is set to %f' % (node_exposuretimelowerlimit.GetDisplayName(), node_exposuretimelowerlimit.GetValue()))

        node_exposuretimeupperlimit = PySpin.CFloatPtr(nodemap.GetNode('AutoExposureExposureTimeUpperLimit'))
        if not self.check_available_writable(node_exposuretimeupperlimit): return False
        node_exposuretimeupperlimit.SetValue(self.exposuretimeupperlimit)
        self.update_log(
            '%s is set to %f' % (node_exposuretimeupperlimit.GetDisplayName(), node_exposuretimeupperlimit.GetValue()))

        # self.reset_exposure()

        return

    def close(self):
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        if hasattr(self, 'cam'):
            self.cam.DeInit()
            del self.cam
        # Clear camera list before releasing system
        self.cam_list.Clear()

        # Release system instance
        self.system.ReleaseInstance()
        self.update_log('Camera closed...')

    def start_continue(self):
        # :param cam: Camera to run on.
        # :type cam: CameraPtr
        # :return: True if successful, False otherwise.
        # :rtype: bool
        if self.flag_continue:
            self.update_log('Acquisition already started...')
            return

        try:
            result = True

            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()

            # Retrieve GenICam nodemap
            nodemap = self.cam.GetNodeMap()

            sNodemap = self.cam.GetTLStreamNodeMap()

            # Change bufferhandling mode to NewestOnly
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
                self.update_log('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
                self.update_log('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve integer value from entry node
            node_newestonly_mode = node_newestonly.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                self.update_log('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                self.update_log('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            self.update_log('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            self.cam.BeginAcquisition()
            self.flag_continue = True

            self.update_log('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                self.update_log('Device serial number retrieved as %s...' % device_serial_number)



        except PySpin.SpinnakerException as ex:
            self.update_log('Error: %s' % ex)
            return False

        return result

    def stop_continue(self):
        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        if self.flag_continue:
            self.flag_continue = False
            self.cam.EndAcquisition()
            self.update_log('Stop acquiring images...')
        else:
            self.update_log('Acquiring already stopped...')

    def acquire_continue(self):
        frames_succ = self.average_frames
        image_data = self.floatzeroframe.copy()
        for i in range(self.average_frames):
            image_result = self.cam.GetNextImage(self.exposuretimeupperlimit)  # GetNextImage( grabTimeout )
            if image_result.IsIncomplete():
                self.update_log('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                image_result.Release()
                frames_succ -= 1
            else:
                image_data += image_result.GetNDArray().astype(float)
                image_result.Release()

        if frames_succ == 0:
            return
        image_data = image_data / frames_succ - self.background
        image_data[image_data < 0] = 0
        self.frame = image_data.astype(uint8)

        #  Ensure image completion
        # if image_result.IsIncomplete():
        #     self.update_log('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        #     image_result.Release()
        #
        # else:
        #     # Getting the image data as a numpy array
        #     image_data = image_result.GetNDArray()
        #     if self.average_frames > 1:
        #         image_data = image_data.astype(float) / self.average_frames
        #         for i in range(self.average_frames - 1):
        #             image_data += image_result.GetNDArray().astype(float) / self.average_frames
        #         image_data = image_data.astype(uint8)
        #
        #     temp_background = self.background
        #     badpoints = temp_background > image_data
        #     temp_background[badpoints] = image_data[badpoints]
        #     self.frame = image_data - temp_background

        #  Release image
        #
        #  *** NOTES ***
        #  Images retrieved directly from the camera (i.e. non-converted
        #  images) need to be released in order to keep from filling the
        #  buffer.

        return

    def set_average_frames(self, average_frames_str):
        try:
            self.average_frames = max(1, int(average_frames_str))
        except ValueError as ex:
            self.update_log('ValueError: %s' % ex)
            return False
        self.update_log('Average frame number : %d'%(self.average_frames))
        return True

    def configure_exposure(self, exposure_time_str):
        """
         This function configures a custom exposure time. Automatic exposure is turned
         off in order to allow for the customization, and then the custom setting is
         applied.

         :param cam: Camera to configure exposure for.
         :type cam: CameraPtr
         :return: True if successful, False otherwise.
         :rtype: bool
        """
        try:
            exposure_time = float(exposure_time_str)
        except ValueError as ex:
            self.update_log('ValueError: %s' % ex)
            return False

        try:
            result = True

            # Turn off automatic exposure mode
            #
            # *** NOTES ***
            # Automatic exposure prevents the manual configuration of exposure
            # times and needs to be turned off for this example. Enumerations
            # representing entry nodes have been added to QuickSpin. This allows
            # for the much easier setting of enumeration nodes to new values.
            #
            # The naming convention of QuickSpin enums is the name of the
            # enumeration node followed by an underscore and the symbolic of
            # the entry node. Selecting "Off" on the "ExposureAuto" node is
            # thus named "ExposureAuto_Off".
            #
            # *** LATER ***
            # Exposure time can be set automatically or manually as needed. This
            # example turns automatic exposure off to set it manually and back
            # on to return the camera to its default state.

            if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                self.update_log('Unable to disable automatic exposure. Aborting...')
                return False

            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.update_log('Automatic exposure disabled...')

            # Set exposure time manually; exposure time recorded in microseconds
            #
            # *** NOTES ***
            # Notice that the node is checked for availability and writability
            # prior to the setting of the node. In QuickSpin, availability and
            # writability are ensured by checking the access mode.
            #
            # Further, it is ensured that the desired exposure time does not exceed
            # the maximum. Exposure time is counted in microseconds - this can be
            # found out either by retrieving the unit with the GetUnit() method or
            # by checking SpinView.

            if self.cam.ExposureTime.GetAccessMode() != PySpin.RW:
                self.update_log('Unable to set exposure time. Aborting...')
                return False

            # Ensure desired exposure time does not exceed the maximum
            exposure_time = min(self.cam.ExposureTime.GetMax(), exposure_time)
            exposure_time = max(self.cam.ExposureTime.GetMin(), exposure_time)
            self.cam.ExposureTime.SetValue(exposure_time)
            self.update_log('Exposure time set to %s us...' % exposure_time)

        except PySpin.SpinnakerException as ex:
            self.update_log('Error: %s' % ex)
            result = False

        return result

    def reset_exposure(self):
        """
        This function returns the camera to a normal state by re-enabling automatic exposure.

        :param cam: Camera to reset exposure on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            result = True

            # Turn automatic exposure back on
            #
            # *** NOTES ***
            # Automatic exposure is turned on in order to return the camera to its
            # default state.

            if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                self.update_log('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
                return False

            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

            self.update_log('Automatic exposure enabled...')

        except PySpin.SpinnakerException as ex:
            self.update_log('Error: %s' % ex)
            result = False

        return result

    def get_exposure(self):
        return self.cam.ExposureTime()

    def get_temperature(self):
        self.device_temperature = self.cam.DeviceTemperature()
        return

    def set_background(self):
        self.background = self.frame

    def clear_background(self):
        self.background = self.nobackground

    def update_log(self, log):
        self.log.insertPlainText(log)
        self.log.insertPlainText('\n')

    def check_available_writable(self, node):
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            self.update_log('Unable to set %s. Aborting...' % (node.GetName()))
            return False

        return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    flir = FlirCamController()
    flir.start_continue()
    flir.acquire_continue()
    flir.acquire_continue()
    flir.acquire_continue()
    plt.imshow(flir.frame)
    plt.show()
    flir.stop_continue()
    flir.close()
