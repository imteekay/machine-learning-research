# np_baseball is available

# Import numpy
import numpy as np

# get only height data (first column)
np_height = np_baseball[:,0]

# Print mean height (first column)
avg = np.mean(np_height)
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_height)
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_height)
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
np_weight = np_baseball[:,1]
corr = np.corrcoef(np_height, np_weight)
print("Correlation: " + str(corr))
