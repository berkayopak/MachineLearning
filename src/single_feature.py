import matplotlib.pyplot as plt
import numpy as np

my_data = np.genfromtxt('../data/data.csv', delimiter=',') # data okuma
X = my_data[:, 0].reshape(-1,1) # -1 numpy nin dimension kendi kendine belirlemesini sağlar
ones = np.ones([X.shape[0], 1]) # birler arrayı
X = np.concatenate([ones, X],1) # birleri X matrisi ile birleştir
y = my_data[:, 1].reshape(-1,1) # y matrisi yaratımı

plt.scatter(my_data[:, 0].reshape(-1,1), y)
plt.show()

# hyper parametreler
alpha = 0.0001
iters = 1000

# theta bir satır vektörüdür
theta = np.array([[1.0, 1.0]])

def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2) # @ matrisler ile dizilerin çarpımını sağlar. * kullanmak istiyorsak arrayleri matrise çevirmeliyiz
    return np.sum(inner) / (2 * len(X))

computeCost(X, y, theta)

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        if i % 10 == 0: # her 10 döngüde cost ekrana basılıyor
            print(cost)
    return (theta, cost)

g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g, cost)

plt.scatter(my_data[:, 0].reshape(-1,1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals #doğru denklemi (the line equation)
plt.plot(x_vals, y_vals, '--')

plt.show()