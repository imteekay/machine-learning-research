# height is available as a regular list
height = [43, 53, 65, 54, 62, 99]

# Import numpy
import numpy as np

# Create a numpy array from height: np_height
np_height = np.array(height)

# Print out np_height
print(np_height)

# Convert np_height to m: np_height_m
to_meters = 0.0254
np_height_m = np_height * to_meters

# Print np_height_m
print(np_height_m)
