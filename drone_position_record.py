# Author: Trevor Sherrard
# Since: March 24th, 2022
# Project: UAV Based Disaster Relief System
# Purpose: Allows user to capture current position of the drone
#          and save to CSV

import airsim
import csv
vehicle_name='Drone0'

client = airsim.MultirotorClient()
client.confirmConnection()

client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
# future add ignore collisions until drone is at home position
client.moveToPositionAsync(0, 0, -10, 5,
                           vehicle_name=vehicle_name).join()
client.hoverAsync(vehicle_name=vehicle_name).join()

# open file
file_name = "data/intersection_positions.csv"
file_obj = open(file_name, "w")

# make csv writer
csvwriter = csv.writer(file_obj)

while(True):
    xy = input("X,Y,Z values: ")
    x,y,z=xy.split(',')


    # get status
    state = client.getMultirotorState()
    new_x = state.kinematics_estimated.position.x_val+int(x)
    new_y = state.kinematics_estimated.position.y_val+int(y)
    new_z = state.kinematics_estimated.position.z_val-int(z)

    client.moveToPositionAsync(new_x, new_y,new_z,5,vehicle_name=vehicle_name).join()

    client.hoverAsync(vehicle_name=vehicle_name).join()

    # write row
    row = [new_x, new_y, new_z]

    csvwriter.writerow(row)
