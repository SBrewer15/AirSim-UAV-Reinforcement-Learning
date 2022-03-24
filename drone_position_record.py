# Author: Trevor Sherrard
# Since: March 24th, 2022
# Project: UAV Based Disaster Relief System
# Purpose: Allows user to capture current position of the drone
#          and save to CSV

import airsim
import csv

client = airsim.MultirotorClient()
client.confirmConnection()

# open file
file_name = "intersection_positions.csv"
file_obj = open(file_name, "w")

# make csv writer
csvwriter = csv.writer(file_obj)

while(True):
    user_input = input("press any key to log current position")

    # get status
    state = client.getMultirotorState()
    new_x = state.kinematics_estimated.position.x_val
    new_y = state.kinematics_estimated.position.y_val
    new_z = state.kinematics_estimated.position.z_val

    # write row
    row = [new_x, new_y, new_z]
    csvwriter.writerow(row)

