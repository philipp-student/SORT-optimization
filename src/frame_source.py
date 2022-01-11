import queue
from skimage import io
import os
import glob
from carla_camera_frame import CARLACameraFrame
import socket
import threading
import struct
import pickle

class FrameSource():
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.frame_queue_lock = threading.Lock()
    
    def get_frame(self, timeout=1):
        try:
            # Acquire lock for frame queue.
            self.frame_queue_lock.acquire()            
            
            # Get frame from queue.
            frame = self.frame_queue.get(True, timeout)                        
            
            # Return frame.
            return frame
        
        except queue.Empty:
            # Return none if queue was empty.
            return None
        
        finally:
            # Release frame queue lock.
            self.frame_queue_lock.release()

    def cleanup(self):
        pass
    
class ServerFrameSource(FrameSource):
    def __init__(self, host, port):
        FrameSource.__init__(self)
        
        # Set TCP connection details.
        self.host = host
        self.port = port
        self.server_connection = None
        
        # Create a TCP socket connection to the server.
        self.server_connection = socket.create_connection((self.host, self.port))
        
        # Setup thread that receives the frames.
        self.frame_receival_thread = threading.Thread(target=self._receive_frames)
        self.thread_cancellation_requested = False
        
        # Start frame receival thread.                
        self.frame_receival_thread.start()  
    
    def cleanup(self):
        # Signal thread stop.
        self.thread_cancellation_requested = True
        
        # Wait for thread to stop execution.
        self.frame_receival_thread.join()
        
        # Close server connection.
        self.server_connection.close()
        
    def _receive_frames(self):        
        while(not self.thread_cancellation_requested):
            # Get frame from server.
            frame = self._receive_frame(self.server_connection)
            
            if not frame:
                print("Error: Frame could not be deserialized")
            else:                
                # Acquire frame queue lock.
                self.frame_queue_lock.acquire()                
                
                # Put frame into queue.
                self.frame_queue.put(frame)
                
                # Release frame queue lock.
                self.frame_queue_lock.release()                    
        
    def _receive_frame(self, server_connection):
        raw_length = self._receive_n(server_connection, 4)
        
        if not raw_length:
            return None
        
        length = struct.unpack('>I', raw_length)[0]
        
        frame_serialized = self._receive_n(server_connection, length)
        
        return pickle.loads(frame_serialized)

    def _receive_n(self, server_connection, n):
        data = bytearray()
        
        while len(data) < n:
            packet = server_connection.recv(n - len(data))
            
            if not packet:
                return None
            
            data.extend(packet)
            
        return data                

class DirectoryFrameSource(FrameSource):    
    def __init__(self, frame_directory):
        FrameSource.__init__(self)
        
        # Set frame directory.
        self.frame_directory = frame_directory
        
        # Load frames from directory.
        self._load_frames(self.frame_directory)
       
    def _load_frames(self, frame_directory):
        # Get all jpg files in frame directory.
        frame_files_regex = os.path.join(frame_directory, '*.jpg')
        frame_files = glob.glob(frame_files_regex)
        
        # Acquire frame queue lock.
        self.frame_queue_lock.acquire()
            
        # Create a camera frame for each image and put it into the queue.
        for frame_file_index, frame_file_name in enumerate(frame_files):          
            # Load frame from file.  
            frame = CARLACameraFrame.from_image(frame_file_name, frame_file_index)                        
            
            # Add frame to queue.
            self.frame_queue.put(frame)
            
        # Release frame queue lock.
        self.frame_queue_lock.release()         