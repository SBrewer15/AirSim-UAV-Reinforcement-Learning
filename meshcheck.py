import airsim
import numpy as np
import pandas as pd
import nb_files.nb_Utilities as util
import time

util.set_seed(42)

client = airsim.MultirotorClient()                                       # connect to the simulator
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone0")                     # enable API control on Drone1
client.armDisarm(True, vehicle_name="Drone0")                            # arm Drone1

client.takeoffAsync(vehicle_name="Drone0").join()                        # let Drone1 take-off
client.moveToPositionAsync(20, 3, -1, 5, vehicle_name="Drone0").join()   # Drone1 moves to (20, 3, 1) at 5m/s and hovers (note the inverted Z axis)
client.hoverAsync(vehicle_name="Drone0").join()

#AIRSIM_HOST_IP='127.0.0.1'

#client = airsim.VehicleClient(ip=AIRSIM_HOST_IP)
#client.confirmConnection()

# List of returned meshes are received via this function
meshes=client.simGetMeshPositionVertexBuffers()

arr=np.array([m.name for m in meshes])
df=pd.DataFrame(arr)
df.to_csv('data/meshes.csv')
print('saved and done...')
