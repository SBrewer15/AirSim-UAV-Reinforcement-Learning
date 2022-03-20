# Author: Trevor Sherrard
# Since: March 20th, 2022
# Project: UAV Based Disaster Relief System
# Purpose: To demonstrate the obtainment of distances sensor data from
#          drones within unreal engine

import airsim

def monitor_dists():
    # make client connection
    client = airsim.MultirotorClient()
    client.confirmConnection()

    while(True):
        # get distance sensor data for drone0
        data_drone_0_front = client.getDistanceSensorData(vehicle_name="Drone0", distance_sensor_name="DistanceFront_Drn0")
        data_drone_0_right = client.getDistanceSensorData(vehicle_name="Drone0", distance_sensor_name="DistanceRight_Drn0")
        data_drone_0_back = client.getDistanceSensorData(vehicle_name="Drone0", distance_sensor_name="DistanceBack_Drn0")
        data_drone_0_left = client.getDistanceSensorData(vehicle_name="Drone0", distance_sensor_name="DistanceLeft_Drn0")

        # get distance sensor data for drone1
        data_drone_1_front = client.getDistanceSensorData(vehicle_name="Drone1", distance_sensor_name="DistanceFront_Drn1")
        data_drone_1_right = client.getDistanceSensorData(vehicle_name="Drone1", distance_sensor_name="DistanceRight_Drn1")
        data_drone_1_back = client.getDistanceSensorData(vehicle_name="Drone1", distance_sensor_name="DistanceBack_Drn1")
        data_drone_1_left = client.getDistanceSensorData(vehicle_name="Drone1", distance_sensor_name="DistanceLeft_Drn1")

        # NOTE: distance returned will be the 'MaxDistance' if nothing is
        # detected between 'MinDistance' and 'MaxDistance'. These values are set in
        # settings.json for each of the distance sensors.
        print("Drone0 Distance Data")
        print("front: {}".format(data_drone_0_front.distance))
        print("right: {}".format(data_drone_0_right.distance))
        print("back: {}".format(data_drone_0_back.distance))
        print("left: {}".format(data_drone_0_left.distance))

        print("Drone1 Distance Data")
        print("front: {}".format(data_drone_1_front.distance))
        print("right: {}".format(data_drone_1_right.distance))
        print("back: {}".format(data_drone_1_back.distance))
        print("left: {}".format(data_drone_1_left.distance))
        print("\n")

if(__name__ == "__main__"):
    monitor_dists()



