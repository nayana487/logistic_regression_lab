# Author: Haley Boyan
# DSI-DC-1 Week 4 Lab 2.2: Logistic Regression
# Perform a logistic regression (logit) on the IRIS dataset using all four features.
'''
Requirements
Practice Statsmodels to run Logistic on the project dataset
Explain how to evaluate a logistic regression model
Explain how to tune the model
Introduce new data to the model and see if it makes a difference

Deliverable
Outline methods and models
Build a logistic regression model
Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps
'''
import statsmodels.api as st

# import the dataset, declare target and features
iris = st.datasets.get_rdataset('iris','datasets')
y = iris.data.Species
X = iris.data.ix[:, 0:4]

# add column of ones for our y-intercept (http://blog.yhat.com/posts/logistic-regression-and-python.html)
X['intercept'] = 1

# make y variables into dummy variables
y = pd.get_dummies(y,prefix='species')

#declare our model
# Perform the Regression using the Fit Method
# Print a summary of the results

for item in y:
    model = st.MNLogit(y[item],X).fit()
    print item
    print model.summary()


'''
Interesting observations:
- You have to run MNLogit because Logit doesn't work with multiple variables
- The algorithm breaks on the setosa data because it's perfectly predictable
- Need to learn: What it's iterating over, what the value returned is, how to interpret table
'''
