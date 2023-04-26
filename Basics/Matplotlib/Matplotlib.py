from matplotlib import pyplot as plt
import numpy as np



x = np.linspace (-10, 10, 2000)
y = 1 / (1 + np.exp (-x))

plt.plot (x, y)

plt.title ("Sigmoid Function")
plt.xlabel ("X Axis")
plt.ylabel ("Y Axis")

plt.show()
