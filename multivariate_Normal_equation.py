import numpy as np 
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd 
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import pylab
from sklearn import ensemble


#read the data 
lines = open('abalone.domain','r')
names = []
for name in lines :
    temp = name.split(":")[0]
    names.append(temp) 
        
df = column_names = pd.read_csv('abalone.data', names = names)
#preprocess the data 

le=LabelEncoder()
le.fit(np.unique(df['sex']))
df.sex=le.transform(df.sex)

#the height == 0 it doesn't make sense... right so fill it with the mean value 
df.loc[df['height'] == 0, 'height'] =df['height'].mean()
df.loc[df['height'] == 0, 'height'] =df['height'].mean()


x = df.loc[:, df.columns != 'rings']

y = df.loc[:, df.columns == 'rings']   



poly = PolynomialFeatures(3)
new_x = poly.fit_transform(x.values)



minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
new_x = minmax_scaler.fit_transform(new_x)


x_train, x_test, y_train, y_test = train_test_split( new_x,y, test_size=.33, random_state=42,shuffle=False)


designeMatrix = np.ones((x_train.shape[0],x_train.shape[1]+1))
for i in range(x_train.shape[1]):    
    designeMatrix[:,i+1] = x_train[:,i]
    

lambda_reg_values = np.linspace(-1,1,1000)
error = [] 
for ii in lambda_reg_values :
    g= ii *np.identity(166)
    model =np.dot(np.dot(np.linalg.inv(np.dot(designeMatrix.transpose(),designeMatrix)-g),designeMatrix.transpose()),y_train)
    
    featuresMatrix = x_test
    featuresMatrix  = np.insert(featuresMatrix ,0,1,axis=1)
    
    result = np.zeros((featuresMatrix[:,i].shape[0]))
    for i in range(model.shape[0]):
        result += model[i]*featuresMatrix[:,i]
    
    result = result.reshape(len(result),1)
#    test_rmse1 =np.sqrt(np.sum((y_test - result)**2)/len(result))
    test_rmse1 =np.sqrt(np.sum((result-y_test)**2)/len(result))
    error.append(test_rmse1[0])

pylab.plot(lambda_reg_values,error)
pylab.figure('2',figsize=(8, 6),dpi=100)
pylab.plot(np.arange(y_test.shape[0]),y_test,'r--',alpha=0.9)

pylab.plot(np.arange(y_test.shape[0]),result, alpha=0.5)

print("normal equation : ",min(error))
print("lambda : ",lambda_reg_values[error.index(min(error))])




