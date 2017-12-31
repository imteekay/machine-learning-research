# type coersion
np.array([True, 1, 2]) + np.array([3, 4, False]) # array([4, 5, 2])

# parse True to Integer 1
np.array([True, 1, 2]) # array([1, 1, 2])

# parse False to Integer 0
np.array([3, 4, False]) # array([3, 4, 0])
