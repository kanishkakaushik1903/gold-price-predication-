#import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
#data collection and processing
#loading data into pandas data frame
gold_data=pd.read_csv('gld_price_data.csv')

#print first five rows from data frames
print(gold_data.head())
#print last five rows from data frames
print(gold_data.tail())

#getting some basic infromation of data
gold_data.info()
#checking number of missing values
print(gold_data.isnull().sum())
 
#getting statistical measure of data 
print(gold_data.describe())
#splitiing the features and target
numeric_columns = gold_data.select_dtypes(include=[np.number]).columns.tolist()
correlation = gold_data[numeric_columns].corr()
#splitiing the features and target
numeric_columns = gold_data.select_dtypes(include=[np.number]).columns.tolist()
correlation = gold_data[numeric_columns].corr()
X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']
#print(gold_data.columns)
print(X)
print(Y)


#spliting into training data and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

#training model:random forest regressor
model=RandomForestRegressor().fit(X_train,Y_train)

#model evalution
#predication data test
test_data_prediction=model.predict(X_test)
print(test_data_prediction)
#R square error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R square error : ",error_score)

#compare the actual value and predicate value in a plot
Y_test=list(Y_test)
plt.plot(Y_test,color='red',label='Actual value')
plt.plot(test_data_prediction,color='green',label='predicted value')
plt.title('Actual value vs predicated value')
plt.xlabel('number of actual value')
plt.ylabel('GLD')
plt.legend()
plt.show()

