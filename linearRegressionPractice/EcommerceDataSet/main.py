#This project performs a linear regression analysis on an e-commerce customer dataset to predict yearly spending based on user behavior metrics such as session length, time on app, time on website, and length of membership. 
#It explores feature importance, evaluates model performance using common error metrics, and visualizes residuals to check assumptions like normality and randomness.
#The analysis provides insights into which factors most strongly influence customer expenditure.




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math as math
import pylab
import scipy.stats as stats

df=pd.read_csv("Ecommerce Customers.csv")
#print(df.describe())

#sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=df)
#plt.show()

#sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df)
#plt.show()

#sns.pairplot(df,kind='scatter',plot_kws={'alpha':0.4})
#plt.show()

#sns.lmplot(x="Length of Membership",y="Yearly Amount Spent",data=df,scatter_kws={'alpha':0.5})
#plt.show()

X=df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']] #predictive variables
y=df['Yearly Amount Spent'] #target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)   #we're hiding 30% of the data and going to train the model on the remaining 70% then we'll check if it performs well on the other 30%
#print(X_train)
#print(X_test)


#training the model
lm=LinearRegression()
lm.fit(X_train,y_train)

#print(lm.coef_)

#lm.coef_ will give you coefficients of all the X values you passed (avg session length, time on app, etc)
#the linear regression graph eqn is y= b0 + (b1x1 + b2x2 + b3x3 +....)
#the coefficients b1,b2,b3 etc give the importance of the predictive variables we passed in giving the target variable

#cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coef'])
#print(cdf)

#this is the output we get for the coefficients
#                           Coef
#Avg. Session Length   25.724256
#Time on App           38.597135
#Time on Website        0.459148
#Length of Membership  61.674732

#length of membership is the most important predictive variable followed by time on app,avg session length and so on

#predictions

predictions=lm.predict(X_test)    #this gives us an array of all the expected values for y_test
#print(predictions)


print("Mean Absolute Error",mean_absolute_error(y_test,predictions))#average distance between original line and the line we've predicted.
print("Mean Squared Error",mean_squared_error(y_test,predictions))
print("Root Mean Squared Error",math.sqrt(mean_squared_error(y_test,predictions)))   

sns.scatterplot(x=predictions,y=y_test)  #this is a diagnostic plot. By placing predictions on the x axis and y_test values on the y axis we can get how close the predictions are to the real values. If the values match exactly the graph will plot a y=x line (45 degree line)
plt.xlabel("Predictions")
plt.title("Evaluation of Customer Expenditure")
plt.show()

#residuals-finding normality
residuals=y_test-predictions
sns.displot(residuals,bins=30,kde=True)
plt.show()

#residuals vs predictions
sns.scatterplot(x=predictions,y=residuals)
plt.axhline(0)
plt.show()

#compare our observed residuals against theoretical quantiles of a normal distribution
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()



