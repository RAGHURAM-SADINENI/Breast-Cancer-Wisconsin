import pandas as pd
import numpy as np
from sklearn import model_selection,tree

df=pd.read_csv('Fdata.csv')
x=df.ix[:,(1,2,3,4,5,6,7,8,9)].values
y=df.ix[:,(10)].values
x_p=x[0]
x_p=x_p.reshape(1,-1)
#print(x,y)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=467)

# Creating the classifier object
clf_gini = tree.DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=1)

# Performing training
clf_gini.fit(x_train, y_train)

clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 1)

# Performing training
clf_entropy.fit(x_train, y_train)

y_pred = clf_entropy.predict(x_p)
print("Predicted values:")
print(y_pred[0])
