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
    client_socket.connect(('localhost', 49957))

    inputs=[client_socket]
    timeout = 0.01
    while True:
        flir.acquire_continue()
        try:
            readable,writable,exceptional = select.select(inputs,[],[],timeout)
            if readable:
                data = client_socket.recv(1024)
                data=data.decode('utf-8')
                print('receive',data)
                print(type(data))
                if 'run' in data:
                    flir.file_save()
                    message= flir.file_path.encode('utf-8')
                    client_socket.sendto(message,('localhost', 49957))
                if 'auto_expo' in data:
                    flir.reset_exposure()
                    print(flir.get_exposure())
                if 'expo_time' in data:
                    expotime=float(data[10:])
                    expotime=expotime*1000
                    print(expotime)

                    flir.configure_exposure(expotime)

            time.sleep(0.1)
        except Exception as e :
            print('error:',e)
            time.sleep(0.1)

    flir.stop_continue()
    flir.close()
    client_socket.close()
save_image()