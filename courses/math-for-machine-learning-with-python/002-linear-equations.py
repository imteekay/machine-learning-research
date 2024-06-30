import pandas as pd
from matplotlib import pyplot as plt

df = pd.DataFrame({'x': range(-10, 11)})

# only displaying the x column (x-axis)
df

# add a y column by applying the solved equation to x
df['y'] = (3 * df['x'] - 4) / 2

# display the dataframe
df

plt.plot(df.x, df.y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

# add axis lines for 0,0
plt.axhline()
plt.axvline()

# annote points when x = 0 and y = 0
plt.annotate('x-intercept', (1.333, 0))
plt.annotate('y-intercept', (0, -2))

# set the slope
m = 1.5

# get the y-intercept
yInt = -2

# plot the slope from the y-intercept for 1x
mx = [0, 1]
my = [yInt, yInt + m]
plt.plot(mx, my, color='red')

# plot the graph 
plt.show()