# Import numpy
import numpy as np

# height and weight are available as a regular lists
height = [43, 53, 65, 54, 62, 99]
weight = [70, 73, 85, 64, 92, 109]

to_height_meters = 0.0254
to_weight_meters = 0.453592

# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * to_height_meters

# Create array from weight with correct units: np_weight_kg
np_weight_kg = np.array(weight) * to_weight_meters

# Calculate the BMI: bmi
bmi = np_weight_kg / (np_height_m * np_height_m)

# Print out bmi
print(bmi)
