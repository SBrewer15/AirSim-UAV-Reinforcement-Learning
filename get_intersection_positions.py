# Author: Trevor Sherrard
# Since: March 31, 2022
# Project: RBE Capstone
# Purpose: Extract intersection locations from road data

import csv
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point

filename = "data/intersection_positions.csv"

# open CSV file and extract x and y position vectors
x_vals = list()
y_vals = list()

with open(filename, newline="") as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    for row in csv_reader:
        x_vals.append(float(row[0]))
        y_vals.append(float(row[1]))

# convert x and y vectors to numpy arrays
x_arr = np.array(x_vals)
y_arr = np.array(y_vals)

plt.plot(x_arr, y_arr)

# plot points of interest
points_of_interest = [(125, 125), (0, 125), 
                      (-120, 125), (-125, 0), 
                      (-125, -125), (0, -125),
                      (125, -125), (125, 0), 
                      (75, 1), (80, -125),
                      (0, 0)]

across_horizontal = LineString([Point(-125, 0), Point(125, 0)])
across_vert = LineString([Point(0, -125), Point(0, 125)])
side_street = LineString([Point(81, -125), Point(75, -1)])
missing_bottom = LineString([Point(125, 125), Point(125, 0)])

plt.plot(*across_horizontal.xy)
plt.plot(*across_vert.xy)
plt.plot(*side_street.xy)
plt.plot(*missing_bottom.xy)

for pt in points_of_interest:
    label_str = "x: {}, y: {}".format(pt[0], pt[1])
    plt.plot(pt[0], pt[1], marker='o',
            markersize=3, color="red",
            label=label_str)
    plt.annotate(label_str, pt)


plt.gca().invert_yaxis()
plt.show()
