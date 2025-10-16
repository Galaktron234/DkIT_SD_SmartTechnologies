import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)
    
    
def sigmoid(score):
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1/m)*(np.log(p).T * y + np.log(1-p).T*(1-y))
    return cross_entropy

n_pts = 100
np.random.seed(0)
b = np.ones(n_pts)
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)
top_region = np.array([random_x1_values, random_x2_values, b]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts) , np.random.normal(6, 2, n_pts), b]).T

all_points = np.vstack((top_region, bottom_region))

w1 = -0.1
w2 = -0.15
b = 0

line_parameters = np.matrix([w1, w2, b]).T
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
# w1x1 + w2x2 + b = 0
x2 = -b/w2 + x1*(-w1/w2)

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()

linear_combination = all_points * line_parameters
probabilities = sigmoid(linear_combination)
print(probabilities)

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)
print(calculate_error(line_parameters, all_points, y))
