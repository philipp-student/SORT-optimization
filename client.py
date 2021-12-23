import socket, pickle
from PIL import Image
import numpy as np
from carla_camera_frame import CARLA_Camera_Frame
import struct

HOST = 'localhost'
PORT = 50007
PRINT_EVERY = 100

def receive_frame(server_connection):
    raw_length = receive_n(server_connection, 4)
    
    if not raw_length:
        return None
    
    length = struct.unpack('>I', raw_length)[0]
    
    frame_serialized = receive_n(server_connection, length)
    
    return pickle.loads(frame_serialized)
    

def receive_n(server_connection, n):
    data = bytearray()
    
    while len(data) < n:
        packet = server_connection.recv(n - len(data))
        
        if not packet:
            return None
        
        data.extend(packet)
        
    return data
    

# Create a TCP socket connections to the server.
s = socket.create_connection((HOST, PORT))
print("Client started. Press Ctrl-C to stop...")

try:
    # Receive data continously.
    count = 0
    while True:
        
        # Receive frame.
        frame = receive_frame(s)

        if not frame:
            print("Frame is empty")
            break
        else:        
            count += 1
            
            # Print receival of frame. 
            if (count % PRINT_EVERY == 0):       
                print("Received frame no. {0}".format(count))
                
except KeyboardInterrupt:
    print("Stopping client...")
    
    # Close the socket.
    s.close()

input('Client stopped successfully. Press enter to exit...')
