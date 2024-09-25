import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./datasets/weight-height.csv",delimiter=",",skip_header=1)
height = data[:,1]
weight = data[:,2]

#  inches to centimeters
height = height * 2.54
#   pounds to kilograms
weight = weight * 0.453592

#   height statistics
print("\nHEIGHT:")
height_median = np.median(height)
height_mean = np.mean(height)
height_std = np.std(height)
height_var = np.var(height)

print("Median: " + str(height_median))
print("Mean: " + str(height_mean))
print("Standard deviation: " + str(height_std))
print("Variance: " + str(height_var))

#   weight statistics
print("\nWEIGHT:")
weight_median = np.median(weight)
weight_mean = np.mean(weight)
weight_std = np.std(weight)
weight_var = np.var(weight)

print("Median: " + str(weight_median))
print("Mean: " + str(weight_mean))
print("Standard deviation: " + str(weight_std))
print("Variance: " + str(weight_var))

plt.hist(height, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.title('Task 7 Height Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()
