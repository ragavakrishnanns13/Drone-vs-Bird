import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load your CSV data into a pandas DataFrame
data = pd.read_csv('reduced_data.csv')

# Assuming the target column is named 'target', and other columns are features
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (very important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)  # You can choose different kernels like 'rbf' for non-linear data

# Train the SVM model
svm_classifier.fit(X_train, y_train)

# Predict using the test data
predictions = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report
print('Classification Report:')
print(classification_report(y_test, predictions))
