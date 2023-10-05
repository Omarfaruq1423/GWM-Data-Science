import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# path to the csv file
file_path = "iris.csv"

# load the data set
iris_data_set = pd.read_csv(file_path)

# seperate features and target variable
x = iris_data_set.drop(columns=['species'])
y = iris_data_set['species']

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# create a KNN classifier
model = KNeighborsClassifier()

# train the model usinf=g the training dataset
model.fit(X_train, y_train)

# predict the spp of iris flowers in the testing data
y_pred = model.predict(X_test)

# evaluate the models performance
print("Accuracy: ")
print(accuracy_score(y_test, y_pred))


# visualize the performance using a confusion matrix and classification report
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report: ")
target_names = iris_data_set['species'].unique()
print(classification_report(y_test, y_pred, target_names=target_names))
