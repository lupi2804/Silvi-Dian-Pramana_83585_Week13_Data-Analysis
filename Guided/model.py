import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv('iris.csv')
iris_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Selecting features and target variable
X = iris_df.drop('SepalLeghtCm', 'SepalWidthCm', 'PetalLeghtCm','PetalWidthCm')
y = iris_df['Species']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Create and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(rf_classifier, 'model/rf_iris_model.joblib')