# import required packages
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.display.max_columns = 99 #Testing

data = pd.read_csv("dataset.data") # Read file into Dataframe
print(data) # TESTING

# Split the data
numCol = len(data.columns)
X = data.values[:, 0:numCol-1]
Y = data.values[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
print("Data has been split")
