import pandas as pd
import numpy as np
from sklearn import neighbors,model_selection

df=pd.read_csv('Fdata.csv')
x=df.ix[:,(1,2,3,4,5,6,7,8,9)].values
y=df.ix[:,(10)].values
x_p=x[0]
x_p=x_p.reshape(1,-1)
#print(x,y)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=467)

c=neighbors.KNeighborsClassifier()
c.fit(x_train,y_train)
accuracy=c.score(x_test,y_test)
y_p=c.predict(x_p)
print(accuracy)
if(y_p==2):
    print('benign')
else:
    print('malignant')
#print(x_p,y_p)
