from p5 import noise, noise_seed
import matplotlib.pyplot as plt

noise_seed(1)

x = [i for i in range(1000)]
y = [noise(i/100, i/100) for i in range(1000)]
plt.plot(x, y)
plt.show()