import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC

df=pd.read_csv('Fdata111.csv')
df['label']=df['label'].map({2:'benign',4:'malignant'})
x=df.ix[:,(1,2,3,4,5,6,7,8)].values
y=df.ix[:,(9)].values
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.25,random_state=489)

c=SVC(kernel='linear')
c.fit(x_train,y_train)
a=c.score(x_test,y_test)
print(a)
