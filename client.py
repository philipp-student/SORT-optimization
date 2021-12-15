import socket, pickle
from PIL import Image
import numpy as np
from carla_camera_frame import CARLA_Camera_Frame

HOST = 'localhost'
PORT = 50007

# Create a TCP socket connection.
s = socket.create_connection((HOST, PORT))

# Load image.
image = Image.open('./img/test_image.jpg')

# Convert to numpy array.
image = np.array(image)

# Create an instance of the class.
variable = CARLA_Camera_Frame(image, 100)

# Pickle the object.
data_string = pickle.dumps(variable)
length = len(data_string)

# Send the data to the server.
for i in range(10):    
    s.send(data_string)

input('Data Sent to Server. Press enter to exit.')

# Close the socket.
s.close()
