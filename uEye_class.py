from pyueye import ueye
import numpy as np
import cv2
import sys
import ctypes
from time import time #time_ns
import matplotlib.pyplot as plt
import logging

error_codes = {ueye.IS_INVALID_EXPOSURE_TIME: "Invalid exposure time",
               ueye.IS_INVALID_CAMERA_HANDLE: "Invalid camera handle",
               ueye.IS_INVALID_MEMORY_POINTER: "Invalid memory pointer",
               ueye.IS_INVALID_PARAMETER: "Invalid parameter",
               ueye.IS_IO_REQUEST_FAILED: "IO request failed",
               ueye.IS_NO_ACTIVE_IMG_MEM: "No active IMG memory",
               ueye.IS_NO_USB20: "No USB2",
               ueye.IS_NO_SUCCESS: "No success",
               ueye.IS_NOT_CALIBRATED: "Not calibrated",
               ueye.IS_NOT_SUPPORTED: "Not supported",
               ueye.IS_OUT_OF_MEMORY: "Out of memory",
               ueye.IS_TIMED_OUT: "Timed out",
               ueye.IS_SUCCESS: "Success",
               ueye.IS_CANT_OPEN_DEVICE: "Cannot open device",
               ueye.IS_ALL_DEVICES_BUSY: "All device busy",
               ueye.IS_TRANSFER_ERROR: "Transfer error"}

class uEyeException(Exception):
    def __init__(self, error_code):
        self.error_code = error_code

    def __str__(self):
        if self.error_code in error_codes.keys():
            return error_codes[self.error_code]
        else:
            for att, val in ueye.__dict__.items():
                if att[0:2] == "IS" and val == self.error_code \
                        and ("FAILED" in att or
                             "INVALID" in att or
                             "ERROR" in att or
                             "NOT" in att):
                    return "Err: {} ({} ?)".format(str(self.error_code),
                                                   att)
            return "Err: " + str(self.error_code)


def check(error_code):
    """
    Check an error code, and raise an error if adequate.
    """
    if error_code != ueye.IS_SUCCESS:
        raise uEyeException(error_code)

class Camera(object):

    def __init__(self):
        # Variables
        self.hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 1  # 3 channels for color mode (RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.current_fps = None



    def configure(self):
        self.connect()
        self.get_data()
        self.sensor_info()

        self.reset_camera()
        self.set_display_to_DIB()
        self.set_color_mode()
        self.set_full_auto() # i donnot want set allfeatures in auto

    def connect(self):
        # Starts the driver and establishes the connection to the camera
        print("Connecting to camera")
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
        else:
            print("Camera initialised")

    def get_data(self):
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure
        # that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
        # else:
        #     print("GetCameraInfo complete")
            # print(self.cInfo)  # self.cInfo contains a lot of interesting device information
            # print()

    def sensor_info(self):
        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")
        else:
            # print("Get Sensor info complete")
            # print(self.sInfo)
            print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
            print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))

        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_EXTERNALTRIGGER))
        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_SUPPORTED_TRIGGER_MODE))
        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_TRIGGER_STATUS))

    def reset_camera(self):
        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")
        else:
            print("Camera reset complete")

    def set_display_to_DIB(self):
        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetDisplayMode ERROR")
        else:
            print("Camera set to Device Independent Bitmap (DIB) display mode")

    def set_color_mode(self):
        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")


    def det_area_of_interest(self, x=0, y=0, width=1280, height=1024): # AOI should do  more research
    #def det_area_of_interest(self):  # AOI should do  more research
        # set the size and position of an "area of interest"(AOI) within an image

        width = width or self.rectAOI.s32Width
        height = height or self.rectAOI.s32Height
        self.rectAOI.s32X = ueye.int(x)
        self.rectAOI.s32Y = ueye.int(y)
        self.rectAOI.s32Width = ueye.int(width)
        self.rectAOI.s32Height = ueye.int(height)

        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")


        print("Image width:\t", width)
        print("Image height:\t", height)

    def set_full_auto(self):
        print("Setting mode to full auto")
        disable = ueye.DOUBLE(0)
        enable = ueye.DOUBLE(1)
        zero = ueye.DOUBLE(0)
        ms = ueye.DOUBLE(20)
        rate = ueye.DOUBLE(50)
        newrate = ueye.DOUBLE()
        number = ueye.UINT()

        ret = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, enable, zero)
        print('AG:',ret)
        ret = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable, zero)
        print('A_SHUTTER:',ret)
        ret = ueye.is_SetFrameRate(self.hCam, rate, newrate)
        print('FR:',ret,newrate)
        ret = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms, ueye.sizeof(ms))
        print('EXP:',ret,ms)



    def get_parameters_range(self):
        pixelclock = self.get_pixelclock() # dtep1 read pixel clock, change it
        print("pixelclock: ", pixelclock)
        fps_range = self.get_fps_range()
        print("fps_range: min_value {min} max_value {max}".format(min=fps_range[0], max=fps_range[1]) )
        exposure_range = self.get_exposure_range()
        print("exposure_range: min_value {min} max_value {max}".format(min=exposure_range[0], max=exposure_range[1]))



    def set_fps(self, fps):
        """
        Set the fps.

        Returns
        =======
        fps: number
            Real fps, can be slightly different than the asked one.
        """
        # checking available fps
        mini, maxi = self.get_fps_range()
        if fps < mini:
            print(f'Warning: Specified fps ({fps:.2f}) not in possible range:'
                  f' [{mini:.2f}, {maxi:.2f}].'
                  f' fps has been set to {mini:.2f}.')
            fps = mini
        if fps > maxi:
            print(f'Warning: Specified fps ({fps:.2f}) not in possible range:'
                  f' [{mini:.2f}, {maxi:.2f}].'
                  f' fps has been set to {maxi:.2f}.')
            fps = maxi
        fps = ueye.c_double(fps)
        new_fps = ueye.c_double()
        check(ueye.is_SetFrameRate(self.hCam, fps, new_fps))
        self.current_fps = float(new_fps)
        return new_fps

    def get_fps(self):
        """
        Get the current fps.

        Returns
        =======
        fps: number
            Current fps.q
        """
        if self.current_fps is not None:
            return self.current_fps
        fps = ueye.c_double()
        check(ueye.is_GetFramesPerSecond(self.hCam, fps))
        print('current_fps {fp} s'.format(fp=1/fps))
        return float(1/fps)



    def get_fps_range(self):
        """
        Get the current fps available range.

        Returns
        =======
        fps_range: 2x1 array
            range of available fps
        """
        mini = ueye.c_double()
        maxi = ueye.c_double()
        interv = ueye.c_double()
        check(ueye.is_GetFrameTimeRange(
                self.hCam,
                mini, maxi, interv))
        return [float(1/maxi), float(1/mini)]

    def set_pixelclock(self, pixelclock):
        """
        Set the current pixelclock.

        Params
        =======
        pixelclock: number
            Current pixelclock.
        """
        # Warning
        print('Warning: when changing pixelclock at runtime, you may need to '
              'update the fps and exposure parameters')
        # get pixelclock range
        pcrange = (ueye.c_uint*3)()
        check(ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                                 pcrange, 12))
        pcmin, pcmax, pcincr = pcrange
        if pixelclock < pcmin:
            pixelclock = pcmin
            print(f"Pixelclock out of range [{pcmin}, {pcmax}] and set "
                  f"to {pcmin}")
        elif pixelclock > pcmax:
            pixelclock = pcmax
            print(f"Pixelclock out of range [{pcmin}, {pcmax}] and set "
                  f"to {pcmax}")
        # Set pixelclock
        pixelclock = ueye.c_uint(pixelclock)
        check(ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET,
                                 pixelclock, 4))

    def get_pixelclock(self):
        """
        Get the current pixelclock.

        Returns
        =======
        pixelclock: number
            Current pixelclock.
        """
        pixelclock = ueye.c_uint()
        check(ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET,
                                 pixelclock, 4))
        return pixelclock

    def set_exposure(self, exposureTime):
        """
        set exposureTime  !!! when the exposure time is set much higher, it will not change the exposure time in iDS max 87.2
        :param exposureTime: int
        :return:
        """
        new_exposure = ueye.c_double(exposureTime)
        nRet = ueye.is_Exposure(self.hCam,
                                ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,  # in ms
                                new_exposure, 8)
        if nRet != ueye.IS_SUCCESS:
           print("set_exposure: is_CaptureVideo ERROR")

    def get_exposure(self):
        # read the exposure time in the iDS camera
        exposure = ueye.c_double()
        # exposure = ctypes.c_double() # import ctypes
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                         exposure, 8) # ???questions: what is 8
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")
        print('exposure {exp} ms'.format(exp=exposure))
        return exposure



    def get_exposure_range(self):
        param1 = ctypes.c_double(0)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, param1,8)#8
        param2 = ctypes.c_double(0)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, param2,8)#8
        #rec=ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_CAPS,param,4)
        return [param1.value / 1000, param2.value / 1000]  # seconds


    def set_exposure_auto(self, toggle):
        """
        Set auto expose to on/off.

        Params
        =======
        toggle: integer
            1 activate the auto gain, 0 deactivate it
        """
        value = ueye.c_double(toggle)
        value_to_return = ueye.c_double()
        check(ueye.is_SetAutoParameter(self.h_cam,
                                       ueye.IS_SET_ENABLE_AUTO_SHUTTER,
                                       value,
                                       value_to_return))

    def set_gain_auto(self, toggle):
        """
        Set/unset auto gain.

        Params
        ======
        toggle: integer
            1 activate the auto gain, 0 deactivate it
        """
        value = ueye.c_double(toggle)
        value_to_return = ueye.c_double()
        check(ueye.is_SetAutoParameter(self.h_cam,
                                       ueye.IS_SET_ENABLE_AUTO_GAIN,
                                       value,
                                       value_to_return))

    def __get_timeout(self):
        fps = self.get_fps()
        if fps == 0:
            fps = 1
        return int(1.5*(1/fps)+1)*1000

    def allocate_memory_for_image(self):
        # Allocates an image memory for an image having its dimensions defined by width and height
        # and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.rectAOI.s32Width, self.rectAOI.s32Height, self.nBitsPerPixel,
                                     self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

    def set_external_trigger(self):
        # trigger mode hi_lo triggers on falling edge
        trigmode = ueye.IS_SET_TRIGGER_HI_LO
        nRet = ueye.is_SetExternalTrigger(self.hCam, trigmode)
        if nRet != ueye.IS_SUCCESS:
            print("SetExternalTrigger ERROR")
        print('External trigger mode set', ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_EXTERNALTRIGGER), trigmode)

    def set_trigger_counter(self, nValue):
        return ueye.is_SetTriggerCounter(self.hCam, nValue)

    def capture_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.hCam, wait_param)

    def queue_mode(self):
        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.rectAOI.s32Width,
                                       self.rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

    def stop_video(self):
        return ueye.is_StopLiveVideo(self.hCam, ueye.IS_FORCE_VIDEO_STOP)

    def freeze_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_FreezeVideo(self.hCam, wait_param)

    def take_image(self, display=True, threshold2=200):
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        array = ueye.get_data(self.pcImageMemory, self.rectAOI.s32Width, self.rectAOI.s32Height, self.nBitsPerPixel,
                              self.pitch, copy=False)

        t_s = time() / 1e9 #time_ns  # question:unit
        # print('array',np.size(array))
        # bytes_per_pixel = int(nBitsPerPixel / 8)

        # ...reshape it in an numpy array...
        frame = np.reshape(array, (self.rectAOI.s32Height.value, self.rectAOI.s32Width.value, self.bytes_per_pixel))
        # ...resize the image by a half
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Include image data processing here: in this case, use Canny edge detection to find the edge
        # edges = cv2.Canny(frame, 0, threshold2)
        # here's one value for the edge, at height '10'
        # print(np.argmax(edges[10]))
        # here's an average over all horizontal slices.
        # print(time_ns(), np.mean([np.argmax(a) for a in edges]))
        # better still, take the median value
        #dp = t_s, np.median(([np.argmax(a) for a in edges]))
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # ...and finally display it
        if display:

            cv2.imshow("SimpleLive_Python_uEye_OpenCV", frame)
        #cv2.imshow("Edges", edges)

        return t_s



    def activate_live_video(self):
        # Activates the camera's live video mode (free run mode)
        nRet = self.capture_video(wait=False)
        self.get_exposure()  # the current parameter of camera
        #self.get_fps()
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID,
                                       self.rectAOI.s32Width, self.rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Live video activated. Press q to quit")
            #self.get_exposure()

            self.live_video_loop(nRet)
            self.release_memory()

    def live_video_loop(self, nRet):
        # Continuous image display
        while (nRet == ueye.IS_SUCCESS):

            self.take_image()

            # Press q if you want to end the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def save_edge_data(times, edges):
        savepath = r'C:\Users\zhai\PycharmProjects\imagedata/edge-data_{}.csv'.format(times[0])
        with open(savepath, mode='w') as fp:
            fp.write("Timestamp,Edge (pixel)\n")
            for a, b in zip(times, edges):
                fp.write("{:.3f},{}\n".format(a - times[0], b))
            fp.close()
        print("Data saved to {}".format(savepath))

        plt.scatter(times, edges)
        plt.title("Plunger motion")
        plt.xlabel("Time (s)")
        plt.ylabel("Location (pixel)")
        plt.tight_layout()
        plt.savefig(savepath.strip("csv") + "png")
        plt.show()

    def triggered_video(self):
        self.set_external_trigger()
        ueye.is_CaptureVideo(self.hCam, ueye.int(2000))

        self.queue_mode()

        # Continuous image display
        while 0 == ueye.IS_SUCCESS:
            # ueye.is_Event(self.hCam, UINT
            # nCommand, void * pParam, UINT
            # nSizeOfParam)
            try:
                self.freeze_video()
                self.take_image()
                self.release_memory()
            except ValueError:
                continue

            # Press q if you want to end the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_video()
                break

    def release_memory(self):
        # Release the image memory that was allocated using is_AllocImageMem() and remove it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

    def close_connection(self):
        # Destroys the OpenCv windows
        # cv2.destroyAllWindows()
        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)
        print()
        print("END")





if __name__ == "__main__":

    cam = Camera()
    cam.configure()
    cam.det_area_of_interest()
    cam.allocate_memory_for_image()
    # we must adjust the other paramters, to get a appropriate exposure time, the adjusting sequence
    # pixel clock---->framerate------->exposure time
    cam.get_parameters_range() # habe a look at the range of every parameter of camera
       # the current parameter of camera
    #cam.get_fps()
    cam.set_exposure(20)
    cam.activate_live_video()
    #cam.triggered_video()
    cam.close_connection()


