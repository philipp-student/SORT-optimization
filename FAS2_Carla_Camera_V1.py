# Global import stuff for carla
import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import logging
import random

# Importing numpy for data processing and cv2 for image visualization
import numpy as np
import cv2

# Properties of the image being created by the camera
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Function that is being called whenever data from the camera is recieved
def process_img(image):
    # Store the raw image data into numpy array
    i = np.array(image.raw_data)

    # Original shape is a flat vector (1228800,) -> Reshape to get different channels (RGBA)
    i = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))

    # Extract only RGB from RGBA data -> Discart alpha channel
    i = i[:, :, :3] #select entire height, entire width, only first 3 channels (rgb)

    print(i)

    # Display image with openCV
    cv2.imshow("", i)
    cv2.waitKey(1)

    # Saving image to disk - Very slow!
    #image.save_to_disk('~/tutorial/output/%.6d.jpg' % image.frame)

    # Returning image data, but normalized (0 <= value <= 1)
    return i/255.0



def main():
    # Arguments for the script to connect to carla server
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        world = client.get_world()
        ego_vehicle = None
        ego_cam = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_gnss = None
        ego_imu = None

        # --------------
        # Spawn vehicle
        # --------------
        
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)
        print('\nEgo color is set')

        # retrieving spawn points where vehicle can be spawned
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')
        

        # --------------
        # Add a RGB camera sensor to ego vehicle. 
        # --------------
        
        # creating rgb camera
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # Setting camera attributes
        cam_bp.set_attribute("image_size_x",f"{IMAGE_WIDTH}")
        cam_bp.set_attribute("image_size_y",f"{IMAGE_HEIGHT}")
        cam_bp.set_attribute("fov",str(105))

        # creating camera location
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)

        # attachting camera to vehicle
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

        # register function that is called whenever data from the camera is recieved
        ego_cam.listen(lambda data: process_img(data))
   


        # --------------
        # Place spectator in Unreal editor on ego spawning
        # --------------
        
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick() 
        spectator.set_transform(ego_vehicle.get_transform())
       

        # --------------
        # Enable autopilot for ego vehicle
        # --------------
        
        ego_vehicle.set_autopilot(True)
       

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()

    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            ego_vehicle.destroy()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')
