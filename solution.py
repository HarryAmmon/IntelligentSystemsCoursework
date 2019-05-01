# import required packages
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
pd.options.display.max_columns = 99 #Testing

data = pd.read_csv("dataset.data") # Read file into Dataframe
#print(data) # TESTING

# Split the data
numCol = len(data.columns)                                                      # Get the number of columns
X = data.values[:, 0:numCol]                                                    # Get data from all rows for all columns bar the last column
Y = data.values[:,-1]                                                           # Get if spam(1) or not (0) from the last column for all rows

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
print("Data has been split")

mlp = MLPClassifier()                                                           # Create the neural network

mlp.fit(X_train, Y_train)                                                       # Train the neural network
parametersMLP = {                                                               # Define some paramaters
    'solver': ['lbfgs', 'sgd', 'adam'],
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(7,7),(7),(6,6),(5,5,5),(2,2,2,2,2,2),(10),(20)]
}

def paramSearch(model, parameters, cv):
    GS = GridSearchCV(estimator = model, param_grid = parameters, cv=cv)
    GS.fit(X_test,Y_test)
    print("Best parameters are: ",GS.best_params_)
    print("Best score is: ",GS.best_score_)
    print(GS.score(X_test,Y_test))

paramSearch(mlp,parametersMLP,5)
