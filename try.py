from distutils import core
import numpy as np

def cross_entropy(y, y_hat):
    return np.sum([-y_i * np.log(y_hat_i) if y_hat_i != 0 else 0 for y_i, y_hat_i in zip(y, y_hat)])

def cross(y, y_hat, e=1e-7):
    values = -y*np.log(y_hat+e)
    if y.ndim == 1:
        return values.sum()
    else:
        return np.sum(values, axis=1)

y_1 = np.array([0, 0, 0, 1, 0])
y_hat_1 = np.array([0.1, 0.1, 0, 0.8, 0])
y = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
y_hat = np.array([[0.1, 0.1, 0, 0.8, 0], [0.1, 0.1, 0, 0.8, 0]])

# print(cross_entropy(y, y_hat))

print(cross(y_1, y_hat_1))
print(cross(y, y_hat))
