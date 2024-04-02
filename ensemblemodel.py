# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



# Load Iris dataset
data = pd.read_csv("resampled.csv")
X = data.drop('target',axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
svm_linear = SVC(kernel='linear',probability=True, C=0.025, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True,C=1, gamma=0.5, random_state=42)  # Non-linear SVM with RBF kernel

# Create an ensemble using VotingClassifier with 'soft' voting
ensemble_model = VotingClassifier(estimators=[('rf', random_forest), ('adaboost', adaboost), ('svm_linear', svm_linear), ('svm_rbf', svm_rbf)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
predictions = ensemble_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Ensemble Model Accuracy:", accuracy)




