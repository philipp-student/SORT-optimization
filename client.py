import socket, pickle
from PIL import Image
import numpy as np
from carla_camera_frame import CARLA_Camera_Frame

HOST = 'localhost'
PORT = 50007
PRINT_EVERY = 100

# Create a TCP socket connections to the server.
s = socket.create_connection((HOST, PORT))
print("Client started. Press Ctrl-C to stop...")

try:
    # Receive data continously.
    count = 0
    while True:
        
        # Retrieve data.
        data = s.recv(822001)

        if not data:
            print("Data is empty")
            break
        else:        
            count += 1
            
            # Print receival of frame. 
            if (count % PRINT_EVERY == 0):       
                print("Received frame no. {0}".format(count))
            
            # Unpickle data.
            data_variable = pickle.loads(data)
except KeyboardInterrupt:
    print("Stopping client...")
    
    # Close the socket.
    s.close()

input('Client stopped successfully. Press enter to exit...')
