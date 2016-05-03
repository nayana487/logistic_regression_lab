import statsmodels.api as st
iris = st.datasets.get_rdataset('iris', 'datasets')
iris.data['species_num']= iris.data.Species.map({'setosa':0, 'versicolor':1, 'virginica':2})
# Setting the species_num turned out to be unnecessary since MNLogit can handle non-binary categoricals
y = iris.data['Species']
print y.head(3)
x = iris.data[['Sepal.Width', 'Sepal.Length', 'Petal.Length', 'Petal.Width']]
print type(x)

iris.data['intercept']=1.0
iris.data.head()
x.head(3)

mdl = st.MNLogit(iris.data['Species'], iris.data[['Sepal.Width', 'Sepal.Length', 'Petal.Length', 'Petal.Width', 'intercept']])

mdl_fit = mdl.fit(method='powell')

print mdl_fit.summary()

mdl_margeff=mdl_fit.get_margeff()

print mdl_margeff.summary()

mdl_fit.predict([1,2,4,5,1]) # Testing the model
""" This took a lot of doing, but I think we got it to a place where it works. The
biggest factor that was holding us up was choosing the wrong method for the MNLogit.
It defaulted to "newton", which after some super preliminary research has to do with what
the model does when it comes across things that are had to classify"""
