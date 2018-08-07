# defining years and population lists
year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
population = [2.53, 2.8, 3, 4.1, 5.5, 6.1, 7, 7.9]

# Print the last item from year and pop
print(year[-1])
print(population[-1])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, population)

# Display the plot with plt.show()
plt.show()
