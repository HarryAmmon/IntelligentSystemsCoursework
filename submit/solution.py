# Import pandas
import pandas as pd
# Import required sklearn moduels
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Read data into a dataframe object called data
data = pd.read_csv("dataset.csv")

# Seperate the dataframe into data(X) and answer(Y) sets
numCol = len(data.columns) # get the number of columns
X = data.values[:,0:numCol-1] # get data from all rows and all columns apart from the last column
Y = data.values[:,-1] # get data from all rows and the last column only

# Standardise the data
scl = StandardScaler()
scl.fit(X)
scl.transform(X)

# Create the Random Forest Classifer using parameters from the bestParameters.py script
clf = RandomForestClassifier(n_estimators=100,bootstrap=False,max_features="log2",max_depth=50,criterion='entropy')

# Create cross validation using 5 split, with a test size of 30% to help avoid over fitting
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Runs the classifer with the chosen cross validation method and determines an accuracy score
scores = cross_val_score(clf, X, Y, cv=cv,scoring='accuracy')

# Prints all scores and average score
print("Scores obtained from all splits: ",scores)
print("Average score: ",scores.mean())
