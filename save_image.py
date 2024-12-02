import select
import  sys
print(sys.prefix)
from FlirCamController_slm import FlirCamController

from FlirCamFake import FakeCamController
import qdarkstyle

import socket
import struct
import time
import math
from PyQt5 import QtWidgets
def save_image():
    print('qidong')
    flir = FlirCamController()
    flir.initialize()
    flir.start_continue()
    flir.set_average_frames(1)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 49958))#49957

    inputs=[client_socket]
    timeout = 0.01
    while True:

        try:
            readable,writable,exceptional = select.select(inputs,[],[],timeout)
            if readable:
                data = client_socket.recv(1024)
                data=data.decode('utf-8')

                if 'run' in data:
                    flir.acquire_continue()
                    flir.file_save(0)
                    message= flir.file_path.encode('utf-8')
                    client_socket.sendto(message,('localhost', 49958))
                if 'take' in data:
                    flir.acquire_continue()
                    data_=int(data[4:])
                    message = b'acquire finish'
                    client_socket.sendto(message, ('localhost', 49958))
                    flir.file_save(data_)
                    message = b'save finish'
                    client_socket.sendto(message, ('localhost', 49958))
                if 'auto_expo' in data:
                    flir.reset_exposure()
                    print(flir.get_exposure())
                    message = str(flir.get_exposure())#.encode('utf-8')
                    message=message.encode('utf-8')
                    client_socket.sendto(message, ('localhost', 49958))
                if 'expo_time' in data:
                    expotime=float(data[10:])
                    expotime=expotime*1000
                    print(expotime)
                    flir.configure_exposure(expotime)
                if 'frameset' in data:
                    flir.stop_continue()
                    data = data[9:]
                    i = 0
                    while data[i] != ',':
                        i = i + 1
                    print(data[:i])
                    offsetx = int(data[:i])
                    data = data[i + 1:]
                    print(data)
                    i = 0
                    while data[i] != ',':
                        i = i + 1
                    offsety = int(data[:i])
                    data = data[i + 1:]
                    print(data)
                    i = 0
                    while data[i] != ',':
                        i = i + 1
                    width = int(data[:i])
                    data = data[i + 1:]
                    print(data)
                    i = 0
                    while data[i] != ',':
                        i = i + 1
                        if i == len(data):
                            break
                    Height = int(data[:i])
                    print(offsetx, offsety, width, Height)
                    flir.setSize(offsetx, offsety, width, Height)
                    flir.start_continue()
            time.sleep(0.01)
        except Exception as e :
            print('error_image_save:',e)
            time.sleep(0.01)

    flir.stop_continue()
    flir.close()
    client_socket.close()
save_image()