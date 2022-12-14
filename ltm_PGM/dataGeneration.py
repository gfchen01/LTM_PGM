import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=2048)

mu1 = np.array([18.0, 12.0])
mu2 = np.array([1.0, 10.0])
mu3 = np.array([12.0, 5.0])

sigma1 = np.array([[7.5, -2.5], [-2.5, 4.5]])
sigma2 = np.array([[4.5, 1.6], [1.6, 6.6]])
sigma3 = np.array([[8.2, -2.5], [-2.5, 6.0]])

data1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=3000)
data2 = np.random.multivariate_normal(mean=mu2, cov=sigma2, size=3000)
data3 = np.random.multivariate_normal(mean=mu3, cov=sigma3, size=4000)

data1 = np.hstack([data1, 0.0 * np.ones((data1.shape[0], 1))])
data2 = np.hstack([data2, 1.0 * np.ones((data2.shape[0], 1))])
data3 = np.hstack([data3, 2.0 * np.ones((data3.shape[0], 1))])

dataY1 = (5.0 * data1[:, 0] * np.sin(data1[:, 1])).reshape((-1, 1))
dataY2 = (data2[:, 0] + data2[:, 1] * data2[:, 1]).reshape((-1, 1))
dataY3 = (data3[:, 0] * data3[:, 1]).reshape((-1, 1))





data1XY = np.hstack((data1, dataY1))
data2XY = np.hstack((data2, dataY2))
data3XY = np.hstack((data3, dataY3))

dataForRegreesion = pd.DataFrame(data=np.vstack((data1XY, data2XY, data3XY)))
dataForRegreesion.to_csv('testSampleForRegression.csv', sep=',')


data1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=1000)
data2 = np.random.multivariate_normal(mean=mu2, cov=sigma2, size=1000)
data3 = np.random.multivariate_normal(mean=mu3, cov=sigma3, size=2000)

data1 = np.hstack([data1, 0.0 * np.ones((data1.shape[0], 1))])
data2 = np.hstack([data2, 1.0 * np.ones((data2.shape[0], 1))])
data3 = np.hstack([data3, 2.0 * np.ones((data3.shape[0], 1))])

# data1Kappa1 = np.array([[1, -1, 1]]).transpose()
# data1Kappa2 = np.array([[1, 1, -1]]).transpose()
# data1Kappa3 = np.array([[-1, 1, 2]]).transpose()

dataY1 = (5.0 * data1[:, 0] * np.sin(data1[:, 1])).reshape((-1, 1))
dataY2 = (data2[:, 0] + data2[:, 1] * data2[:, 1]).reshape((-1, 1))
dataY3 = (data3[:, 0] * data3[:, 1]).reshape((-1, 1))


data1XY = np.hstack((data1, dataY1))
data2XY = np.hstack((data2, dataY2))
data3XY = np.hstack((data3, dataY3))

dataForRegreesion = pd.DataFrame(data=np.vstack((data1XY, data2XY, data3XY)))
dataForRegreesion.to_csv('testData.csv', sep=',')



# totalData = pd.DataFrame(data=np.concatenate((data1, data2, data3)))
# print(np.concatenate((data1, data2, data3)))
#

plt.figure()
plt.scatter(x=data1[:, 0], y=data1[:, 1], c='#2728d6')
plt.scatter(x=data2[:, 0], y=data2[:, 1], c='#d62728')
plt.scatter(x=data3[:, 0], y=data3[:, 1], c='#1c951b')
plt.show()




