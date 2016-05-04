import statsmodels.api as st
import pandas as pd
import pylab as pl
import numpy as np

# import the dataset, declare target and features
iris = st.datasets.get_rdataset('iris','datasets')
y = iris.data.Species
X = iris.data.ix[:, 0:4]


X = X.as_matrix(columns=None)
y = y.as_matrix(columns=None)

### specify the model, call the final fit, md1_fit, hint use MNLogit from class SM,
# and call the .fit() method on the object
mdl = st.MNLogit(y, X)
mdl_fit = mdl.fit()
# Setup the Logistic Regression
# add column of ones for our y-intercept (http://blog.yhat.com/posts/logistic-regression-and-python.html)

#declare our model

mdl_fit.summary()

# Print a summary of the results
