import socket, pickle
import threading
from PIL import Image
import numpy as np
from carla_camera_frame import CARLA_Camera_Frame
import struct

HOST = 'localhost'
PORT = 50007

# Load image.
image = Image.open('./img/test_image.jpg')

# Convert to numpy array.
TEST_IMAGE = np.array(image)

def send_frame(client_connection, frame):
    # Serialize frame.
    frame_serialized = pickle.dumps(frame)
    
    # Prefix serialized frame with length.
    frame_serialized = struct.pack('>I', len(frame_serialized)) + frame_serialized
    
    # Send frame.
    client_connection.sendall(frame_serialized)

# Holds the connected clients.
connected_clients = []

# Synchronizes the access to the connected clients.
clients_lock = threading.Lock()

# To cancel the informer thread.
thread_cancellation_requested = False

# Thread function that send the received frame to all connected clients.
def inform_clients():
    while(not thread_cancellation_requested):
        
        to_remove = []
        
        clients_lock.acquire()
        
        # If there are any clients connected, send the given frame to all of them.
        if len(connected_clients) != 0:
            for index_current_client, current_client_info in enumerate(connected_clients):                
                # Create an instance of the class.
                variable = CARLA_Camera_Frame(TEST_IMAGE, 100)

                try:
                    # Send the data to the client.
                    send_frame(current_client_info[0], variable)
                except ConnectionAbortedError:
                    print("Client {0} disconnected. (ABORT)".format(current_client_info[1]))
                    
                    # Mark client to be removed.
                    to_remove.append(index_current_client)
                except ConnectionResetError:    
                    print("Client {0} disconnected. (RESET)".format(current_client_info[1]))
                    
                    # Mark client to be removed.
                    to_remove.append(index_current_client)
                
                    
        # Remove clients if any are marked.
        for _, index_client_to_remove in enumerate(to_remove):
            connected_clients.pop(index_client_to_remove)

        clients_lock.release()    

# Create thread that informs the connected clients.
informer_thread = threading.Thread(target=inform_clients)
informer_thread.start()

# Open socket and listen.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("Listening on port", PORT, "...")

while True:
    
    # Wait for a client to connect.
    client_info = s.accept()
    print ('Client connected: ', client_info[1])
    
    # Save client.
    clients_lock.acquire()
    connected_clients.append(client_info)
    clients_lock.release()
