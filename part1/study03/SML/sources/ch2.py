"""
Samples for chapter 2
"""

################# Sample 1 #################
"""
>>> import matplotlib.pyplot as plt
>>> X = [[6], [8], [10], [14],   [18]]
>>> y = [[7], [9], [13], [17.5], [18]]
>>> plt.figure()
>>> plt.title('Pizza price plotted against diameter')
>>> plt.xlabel('Diameter in inches')
>>> plt.ylabel('Price in dollars')
>>> plt.plot(X, y, 'k.')
>>> plt.axis([0, 25, 0, 25])
>>> plt.grid(True)
>>> plt.show()
"""
import matplotlib.pyplot as plt
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()


################# Sample 2 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> # Training data
>>> X = [[6], [8], [10], [14],   [18]]
>>> y = [[7], [9], [13], [17.5], [18]]
>>> # Create and fit the model
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> print 'A 12" pizza should cost: $%.2f' % model.predict([12])[0]
A 12" pizza should cost: $13.68
"""
from sklearn.linear_model import LinearRegression
# Training data
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
# Create and fit the model
model = LinearRegression()
model.fit(X, y)
print 'A 12" pizza should cost: $%.2f' % model.predict([12])[0]


################# Figure 1 #################
"""

"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.figure()
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
print 'A 12" pizza should cost: $%.2f' % model.predict([12])[0]
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
plt.show()


################# Figure 2 #################
"""

"""
from sklearn.linear_model.base import LinearRegression
import matplotlib.pyplot as plt
figure = plt.figure()
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
X2 = [[0], [10], [14], [25]]
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
y3 = [14.25, 14.25, 14.25, 14.25]
y4 = y2 * 0.5 + 5
model.fit(X[1:-1], y[1:-1])
y5 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
plt.plot(X2, y3, 'r-')
plt.plot(X2, y4, 'y-')
plt.plot(X2, y5, 'o-')
plt.show()


################# Polynomial Regression figures #################
"""

"""
__author__ = 'gavin'
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.figure()
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
print 'A 12" pizza should cost: $%.2f' % model.predict([12])[0]
y2 = model.predict(X2)
plt.scatter(X, y)
# plt.plot(X2, y2, 'r-')
import numpy as np
poly = PolynomialFeatures(degree=9)
X_p = poly.fit_transform(X)
print len(X)
xx = np.linspace(0, 26, 1000)
regressor_p = LinearRegression()
regressor_p.fit(X_p, y)
print xx.shape
xx_p = poly.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_p.predict(xx_p), c='r')


################# Sample 3 #################
"""
>>> import numpy as np
>>> print 'Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2)
Residual sum of squares: 1.75
"""
import numpy as np
print 'Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2)


################# Sample 4 #################
"""
>>> from __future__ import division
>>> xbar = (6 + 8 + 10 + 14 + 18) / 5
>>> variance = ((6 - xbar)**2 + (8 - xbar)**2 + (10 - xbar)**2 + (14 - xbar)**2 + (18 - xbar)**2) / 4
>>> print variance
23.2
"""
from __future__ import division
xbar = (6 + 8 + 10 + 14 + 18) / 5
variance = ((6 - xbar)**2 + (8 - xbar)**2 + (10 - xbar)**2 + (14 - xbar)**2 + (18 - xbar)**2) / 4
print variance


################# Sample 5 #################
"""
>>> import numpy as np
>>> print np.var([6, 8, 10, 14, 18], ddof=1)
23.2
"""
import numpy as np
print np.var([6, 8, 10, 14, 18], ddof=1)


################# Sample 6 #################
"""
>>> xbar = (6 + 8 + 10 + 14 + 18) / 5
>>> ybar = (7 + 9 + 13 + 17.5 + 18) / 5
>>> cov = ((6 - xbar) * (7 - ybar) + (8 - xbar) * (9 - ybar) + (10 - xbar) * (13 - ybar) +
>>>        (14 - xbar) * (17.5 - ybar) + (18 - xbar) * (18 - ybar)) / 4
>>> print cov
>>> import numpy as np
>>> print np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1]
22.65
22.65
"""
xbar = (6 + 8 + 10 + 14 + 18) / 5
ybar = (7 + 9 + 13 + 17.5 + 18) / 5
cov = ((6 - xbar) * (7 - ybar) + (8 - xbar) * (9 - ybar) + (10 - xbar) * (13 - ybar) +
    (14 - xbar) * (17.5 - ybar) + (18 - xbar) * (18 - ybar)) / 4
print cov
import numpy as np
print np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1]


################# Sample 7 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> X = [[6], [8], [10], [14],   [18]]
>>> y = [[7], [9], [13], [17.5], [18]]
>>> X_test = [[8],  [9],   [11], [16], [12]]
>>> y_test = [[11], [8.5], [15], [18], [11]]
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> print 'R-squared: %.4f' % model.score(X_test, y_test)
R-squared: 0.6620
"""
from sklearn.linear_model import LinearRegression
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8],  [9],   [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
model = LinearRegression()
model.fit(X, y)
print 'R-squared: %.4f' % model.score(X_test, y_test)


################# Sample 8 #################
"""
>>> from numpy.linalg import inv
>>> from numpy import dot, transpose
>>> X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> print dot(inv(dot(transpose(X), X)), dot(transpose(X), y))
[[ 1.1875    ]
 [ 1.01041667]
 [ 0.39583333]]
"""
from numpy.linalg import inv
from numpy import dot, transpose
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
print dot(inv(dot(transpose(X), X)), dot(transpose(X), y))


################# Sample 9 #################
"""
>>> from numpy.linalg import lstsq
>>> X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> print lstsq(X, y)[0]
[[ 1.1875    ]
 [ 1.01041667]
 [ 0.39583333]]
"""
from numpy.linalg import lstsq
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
print lstsq(X, y)[0]


################# Sample 10 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
>>> y_test = [[11],   [8.5],  [15],    [18],    [11]]
>>> predictions = model.predict(X_test)
>>> for i, prediction in enumerate(predictions):
>>>     print 'Predicted: %s, Target: %s' % (prediction, y_test[i])
>>> print 'R-squared: %.2f' % model.score(X_test, y_test)
Predicted: [ 10.0625], Target: [11]
Predicted: [ 10.28125], Target: [8.5]
Predicted: [ 13.09375], Target: [15]
Predicted: [ 18.14583333], Target: [18]
Predicted: [ 13.3125], Target: [11]
R-squared: 0.77
"""
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11],   [8.5],  [15],    [18],    [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s, Target: %s' % (prediction, y_test[i])
print 'R-squared: %.2f' % model.score(X_test, y_test)


################# Sample 11 #################
"""
>>> from sklearn.preprocessing import PolynomialFeatures
>>> X_train = [[6], [8], [10], [14], [18]]
>>> X_test = [[6],  [8],   [11], [16]]
>>> featurizer = PolynomialFeatures(degree=2)
>>> X_train = featurizer.fit_transform(X_train)
>>> X_test = featurizer.transform(X_test)
>>> print X_train
>>> print X_test
[[  1   6  36]
 [  1   8  64]
 [  1  10 100]
 [  1  14 196]
 [  1  18 324]]
[[  1   6  36]
 [  1   8  64]
 [  1  11 121]
 [  1  16 256]]
"""
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
X_test = [[6],  [8],   [11], [16]]
featurizer = PolynomialFeatures(degree=2)
X_train = featurizer.fit_transform(X_train)
X_test = featurizer.transform(X_test)
print X_train
print X_test


################# Sample 12 #################
"""
>>> import pandas as pd
>>> df = pd.read_csv('winequality-red.csv', sep=';')
>>> df.describe()

                pH    sulphates      alcohol      quality
count  1599.000000  1599.000000  1599.000000  1599.000000
mean      3.311113     0.658149    10.422983     5.636023
std       0.154386     0.169507     1.065668     0.807569
min       2.740000     0.330000     8.400000     3.000000
25%       3.210000     0.550000     9.500000     5.000000
50%       3.310000     0.620000    10.200000     6.000000
75%       3.400000     0.730000    11.100000     6.000000
max       4.010000     2.000000    14.900000     8.000000

"""
import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
df.describe()


################# Sample 13 #################
"""
>>> import matplotlib.pylab as plt
>>> plt.scatter(df['alcohol'], df['quality'])
>>> plt.xlabel('Alcohol')
>>> plt.ylabel('Quality')
>>> plt.title('Alcohol Against Quality')
>>> plt.show()
"""
import matplotlib.pylab as plt
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()


################# Sample 14 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> import pandas as pd
>>> import matplotlib.pylab as plt
>>> from sklearn.cross_validation import train_test_split

>>> df = pd.read_csv('wine/winequality-red.csv', sep=';')
>>> X = df[list(df.columns)[:-1]]
>>> y = df['quality']
>>> X_train, X_test, y_train, y_test = train_test_split(X, y)

>>> regressor = LinearRegression()
>>> regressor.fit(X_train, y_train)
>>> y_predictions = regressor.predict(X_test)
>>> print 'R-squared:', regressor.score(X_test, y_test)
0.345622479617
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split

df = pd.read_csv('wine/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print 'R-squared:', regressor.score(X_test, y_test)


################# Sample 15 #################
"""
>>> import pandas as pd
>>> from sklearn. cross_validation import cross_val_score
>>> from sklearn.linear_model import LinearRegression
>>> df = pd.read_csv('data/winequality-red.csv', sep=';')
>>> X = df[list(df.columns)[:-1]]
>>> y = df['quality']
>>> regressor = LinearRegression()
>>> scores = cross_val_score(regressor, X, y, cv=5)
>>> print scores.mean(), scores
0.290041628842 [ 0.13200871  0.31858135  0.34955348  0.369145    0.2809196 ]
"""
import pandas as pd
from sklearn. cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv('data/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print scores.mean(), scores


################# Sample 16 #################
"""
>>> import numpy as np
>>> from sklearn.datasets import load_boston
>>> from sklearn.linear_model import SGDRegressor
>>> from sklearn.cross_validation import cross_val_score
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.cross_validation import train_test_split
>>> data = load_boston()
>>> X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
>>> X_scaler = StandardScaler()
>>> y_scaler = StandardScaler()
>>> X_train = X_scaler.fit_transform(X_train)
>>> y_train = y_scaler.fit_transform(y_train)
>>> X_test = X_scaler.transform(X_test)
>>> y_test = y_scaler.transform(y_test)
>>> regressor = SGDRegressor(loss='squared_loss')
>>> scores = cross_val_score(regressor, X_train, y_train, cv=5)
>>> print 'Cross validation r-sqaured scores:', scores
>>> print 'Average cross validation r-squared score:', np.mean(scores)
>>> regressor.fit_transform(X_train, y_train)
>>> print 'Test set r-squared score', regressor.score(X_test, y_test)
Cross validation r-sqaured scores: [ 0.73428974  0.80517755  0.58608421  0.83274059  0.69279604]
Average cross validation r-squared score: 0.730217627242
Test set r-squared score 0.653188093125
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)
regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print 'Cross validation r-sqaured scores:', scores
print 'Average cross validation r-squared score:', np.mean(scores)
regressor.fit_transform(X_train, y_train)
print 'Test set r-squared score', regressor.score(X_test, y_test)


################# Updated poly 1 #################
"""
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.preprocessing import PolynomialFeatures

>>> X_train = [[6], [8], [10], [14],   [18]]
>>> y_train = [[7], [9], [13], [17.5], [18]]
>>> X_test = [[6],  [8],   [11], [16]]
>>> y_test = [[8], [12], [15], [18]]

>>> regressor = LinearRegression()
>>> regressor.fit(X_train, y_train)
>>> xx = np.linspace(0, 26, 100)
>>> yy = regressor.predict(xx.reshape(xx.shape[0], 1))
>>> plt.plot(xx, yy)

>>> quadratic_featurizer = PolynomialFeatures(degree=2)
>>> X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
>>> X_test_quadratic = quadratic_featurizer.transform(X_test)

>>> regressor_quadratic = LinearRegression()
>>> regressor_quadratic.fit(X_train_quadratic, y_train)
>>> xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

>>> plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
>>> plt.title('Pizza price regressed on diameter')
>>> plt.xlabel('Diameter in inches')
>>> plt.ylabel('Price in dollars')
>>> plt.axis([0, 25, 0, 25])
>>> plt.grid(True)
>>> plt.scatter(X_train, y_train)
>>> plt.show()

>>> print X_train
>>> print X_train_quadratic
>>> print X_test
>>> print X_test_quadratic
>>> print 'Simple linear regression r-squared', regressor.score(X_test, y_test)
>>> print 'Quadratic regression r-squared', regressor_quadratic.score(X_test_quadratic, y_test)
[[6], [8], [10], [14], [18]]
[[  1   6  36]
 [  1   8  64]
 [  1  10 100]
 [  1  14 196]
 [  1  18 324]]
[[6], [8], [11], [16]]
[[  1   6  36]
 [  1   8  64]
 [  1  11 121]
 [  1  16 256]]
Simple linear regression r-squared 0.809726797708
Quadratic regression r-squared 0.867544365635
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14],   [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6],  [8],   [11], [16]]
y_test = [[8], [12], [15], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print X_train
print X_train_quadratic
print X_test
print X_test_quadratic
print 'Simple linear regression r-squared', regressor.score(X_test, y_test)
print 'Quadratic regression r-squared', regressor_quadratic.score(X_test_quadratic, y_test)
