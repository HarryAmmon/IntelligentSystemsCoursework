from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv("dataset.csv")
numCol = len(data.columns)

X = data.values[:,0:numCol-1]
Y = data.values[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_true = train_test_split(X,Y,test_size=0.2, random_state = 42)
print("Data has been split")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#print("Data has been scaled")


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=400)
parametersMLP = {
    'solver': ['lbfgs', 'sgd', 'adam'],
    'activation' : ['tanh'],
    'hidden_layer_sizes': [(7,7),(15,2),(10,10,10),(100)],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'warm_start': [True,False]
}

GS = GridSearchCV(estimator = mlp, param_grid=parametersMLP, cv=5)
GS.fit(X_train, Y_train)
print("The best parameters are: ",GS.best_params_)
Y_pred = GS.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_true, Y_pred))
print("END")
