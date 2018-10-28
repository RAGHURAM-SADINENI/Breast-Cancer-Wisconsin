import pandas as pd
import csv
df=pd.read_csv('data.csv')
print(df.shape)
df = df[df.g != '?']
#df['label']=df['label'].map({2:'benign',4:'malignant'})
print(df.shape)

with open('WWdata.csv','w') as s:
    w=csv.writer(s,lineterminator='\n')
    w.writerow(('id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','label'))
    w.writerows(df.ix[:,:].values)
