import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#loading the data
data = pd.read_csv("housing.csv")
print(data.head())

#separating the data into target and feature variables
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = data['Price']

#splitting the dataset into training and testing  sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#instantiate the model
linreg = LinearRegression()

#fit the model
linreg.fit(X_train, y_train)

#make a pickle file of the model
pickle.dump(linreg, open("model.pkl", "wb"))