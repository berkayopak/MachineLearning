import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('../data/home.txt',names=["size","bedroom","price"]) #data okuma
original_data = my_data;

print(my_data.head())

#mean normalization kullanarak dataları normalize ediyoruz
my_data = (my_data - my_data.mean())/my_data.std()

print(my_data.head())

#matrislerin ayarlanması
X = my_data.iloc[:,0:2]
print(X)
ones = np.ones([X.shape[0],1])
print(X)
X = np.concatenate((ones,X),axis=1)
print(X)

y = my_data.iloc[:,2:3].values #.values değerleri pandas.core.frame.DataFrame formatından numpy.ndarray formatına dönüştürüyor
theta = np.array([[1.0, 1.0, 1.0]])

#hyper parametleri ayarla
alpha = 0.001
iters = 4000

#cost hesapla
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


#gradient descent
def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# gd ve cost fonksiyonunu calistirma
g, cost = gradientDescent(X, y, theta, iters, alpha)
print(g)
print("theta")

finalCost = computeCost(X, y, g)
print(finalCost)

#cost cizdirme
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

plt.scatter(X[:,1:2], y)
axes = plt.gca() #eksenler
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals #line denklemi
plt.plot(x_vals, y_vals, '--')
plt.show()