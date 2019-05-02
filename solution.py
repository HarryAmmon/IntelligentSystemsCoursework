# import required packages
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_csv("dataset.data") # Read file into Dataframe
#print(data) # TESTING
# Split the data
numCol = len(data.columns)                                                      # Get the number of columns
X = data.values[:, 0:numCol]                                                    # Get data from all rows for all columns bar the last column
Y = data.values[:,-1]                                                           # Get if spam(1) or not (0) from the last column for all rows

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
print("Data has been split")

mlp = MLPClassifier()                                                           # Create the neural network
rf = RandomForestClassifier();
parametersRF = {
    'n_estimators': [10,20,30,25,100],
    'criterion': ["gini","entropy"]
}
parametersMLP = {                                                               # Define some paramaters
    'solver': ['lbfgs', 'sgd', 'adam'],
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(7,7),(7),(6,6)]
}

def paramSearch(model, parameters, cv):
    GS = GridSearchCV(estimator = model, param_grid = parameters, cv=cv)
    #Y_pred = GS.predict(X_test)
    GS.fit(X_train,Y_train)
    return(GS)
    #print("Best score is: ",GS.best_score_)
    #print("Accuracy score is: ", accuracy_score(Y_test,Y_pred))
    #print(GS.score(X_test,Y_test))

params = paramSearch(rf,parametersRF,5)
Y_pred = params.predict(X_test)
if len(Y_test) == len(Y_pred):
    for result, answer in zip(Y_pred, Y_test):
        if result != answer:
            print("Doesn't match")
else:
    print("Not the same length which is odd")
acc = accuracy_score(Y_test,Y_pred)
print(acc)
print("These paramaters where used: ",params.best_params_)
