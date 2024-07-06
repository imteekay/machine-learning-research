from matplotlib import pyplot as plt

# get the extremes for number of chips
chipsAll10s = [16, 0]
chipsAll25s = [0, 16]

# get the extremes for values
valueAll10s = [25, 0]
valueAll25s = [0, 10]

# plot the lines
plt.plot(chipsAll10s, chipsAll25s, color='blue')
plt.plot(valueAll10s, valueAll25s, color="orange")
plt.xlabel('x (£10 chips)')
plt.ylabel('y (£25 chips)')
plt.grid()

plt.show()
