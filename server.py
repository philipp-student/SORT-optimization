import socket, pickle

# Open socket and listen.
HOST = 'localhost'
PORT = 50007

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

# Let the client connect.
conn, addr = s.accept()
print ('Connected by', addr)

count = 0
while True:
    
    # Retrieve data.
    data = conn.recv(822001)

    if not data:
        print("Data is empty")
        break
    else:
        
        count += 1
        print ("Received frame no.{0}".format((count)))
        
        # Unpickle data.
        data_variable = pickle.loads(data)
    
        # Save received frame.
        data_variable.save_frame('received_frame_{0}.png'.format(count))

# Close connection.
conn.close()
