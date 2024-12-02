import select
from FlirCamController_slm import FlirCamController
import  os
import  datetime
import socket
import struct
import time
import math

def save_image():
    print('qidong')
    flir = FlirCamController()
    flir.initialize()
    flir.start_continue()
    flir.set_average_frames(1)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 51814))#49957

    inputs=[client_socket]
    timeout = 0.01
    while True:

        if 1:
            readable,writable,exceptional = select.select(inputs,[],[],timeout)
            if readable:
                data = client_socket.recv(1024)
                data=data.decode('utf-8')

                if 'run' in data:
                    if len(data)>4:
                        now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\image_intensity'
                        name = f"image_{now}.bmp"
                        path = os.path.join(directory, name)
                    else:
                        now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\image0'
                        name = f"image_{now}.bmp"
                        path = os.path.join(directory, name)

                    flir.acquire_continue()


                    flir.file_save(path)
                    message= path.encode('utf-8')
                    client_socket.sendto(message,('localhost', 49958))
                if 'take' in data:
                    flir.acquire_continue()
                    data_=data[4:]
                    if data_ == '1':
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\image1'
                        now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
                    if data_ == '2':
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\image2'
                        now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
                    if data_ == '3':
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\image3'
                        now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
                    if len(data_)>1:
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction1'
                        now = data_

                    name = f"image_{now}.bmp"
                    path = os.path.join(directory, name)
                    message = b'acquire finish'
                    client_socket.sendto(message, ('localhost', 49958))
                    flir.file_save(path)
                    message = b'save finish'
                    client_socket.sendto(message, ('localhost', 49958))
                if 'took' in data:
                    flir.acquire_continue()
                    data_=data[4:]

                    if len(data_)>1:
                        directory = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction2'
                        now = data_

                    name = f"image_{now}.bmp"
                    path = os.path.join(directory, name)
                    message = b'acquire finish'
                    client_socket.sendto(message, ('localhost',51814))
                    flir.file_save(path)
                    message = b'save finish'
                    client_socket.sendto(message, ('localhost', 51814))
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
                if 'disconnect' in data:
                    flir.stop_continue()
                    flir.close()
                    client_socket.close()
                    return 0
            time.sleep(0.01)
        # except Exception as e :
        #     print('error_image_save:',e)
        #     time.sleep(0.01)

    flir.stop_continue()
    flir.close()
    client_socket.close()
save_image()