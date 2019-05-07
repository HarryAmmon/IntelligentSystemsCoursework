# Import pandas
import pandas as pd
# Import required sklearn moduels
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")

# Seperate the dataframe into data(X) and answer(Y) sets
numCol = len(data.columns) # get the number of columns
X = data.values[:,0:numCol-1] # get data from all rows and all columns apart from the last column
Y = data.values[:,-1] # get data from all rows and the last column only

# Split into X and Y train and test sets. Using a test sixe of 30% to avoid overfitting
X_train, X_test, Y_train, Y_true = train_test_split(X,Y,test_size = 0.3, random_state = 42)

# Standardise the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create classifer
clf = RandomForestClassifier()
# Choose parameters to search through
parametersRF = {
    'n_estimators': [10,100,500,1000],
    'bootstrap': [True,False],
    'criterion': ['gini','entropy'],
    'max_features':['auto','sqrt','log2'],
    'max_depth': [4,20,50]
}
# Create the parameter seach passing through the classifer, parameters, number of KFolds to use and the scoring system to use
GS = GridSearchCV(estimator = clf, param_grid=parametersRF, cv=5, scoring="accuracy",n_jobs=-1) # n_jobs = -1 will use all computing power available
GS.fit(X_train, Y_train) # Fit using the train data
print("The best parameters are: ",GS.best_params_) # Outputs the best parameters to use given the train data

Y_pred = GS.predict(X_test) # Predict the outcomne using the test data
print(accuracy_score(Y_true,Y_pred)) # Compare the predictions against the answers and come up with an accuracy score
