import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set display options
pd.set_option('display.colheader_justify', 'center')

def add_features(data):
    # Distance between centers of boxes
    data['CenterDistance'] = np.sqrt((data['XmaxBoxA'] + data['XminBoxA'] - data['XmaxBoxB'] - data['XminBoxB'])**2 +
                                     (data['YmaxBoxA'] + data['YminBoxA'] - data['YmaxBoxB'] - data['YminBoxB'])**2) / 2
    # Area of boxes A and B
    data['AreaBoxA'] = (data['XmaxBoxA'] - data['XminBoxA']) * (data['YmaxBoxA'] - data['YminBoxA'])
    data['AreaBoxB'] = (data['XmaxBoxB'] - data['XminBoxB']) * (data['YmaxBoxB'] - data['YminBoxB'])
    # Aspect ratio of boxes A and B
    data['AspectRatioBoxA'] = (data['XmaxBoxA'] - data['XminBoxA']) / (data['YmaxBoxA'] - data['YminBoxA'])
    data['AspectRatioBoxB'] = (data['XmaxBoxB'] - data['XminBoxB']) / (data['YmaxBoxB'] - data['YminBoxB'])
    # Relative position (left/right)
    data['RelativePosition'] = np.where(data['XminBoxA'] < data['XminBoxB'], 1, 0)
    # Overlap Area
    data['OverlapArea'] = (np.minimum(data['XmaxBoxA'], data['XmaxBoxB']) - np.maximum(data['XminBoxA'], data['XminBoxB'])) * \
                          (np.minimum(data['YmaxBoxA'], data['YmaxBoxB']) - np.maximum(data['YminBoxA'], data['YminBoxB']))
    data['OverlapArea'] = data['OverlapArea'].clip(lower=0)  # Set negative values to 0 (no overlap)
    # Box Size Ratios
    data['BoxSizeRatio'] = data['AreaBoxA'] / data['AreaBoxB']
    # Relative Height Position (1 if A is higher than B, else 0)
    centerY_A = (data['YminBoxA'] + data['YmaxBoxA']) / 2
    centerY_B = (data['YminBoxB'] + data['YmaxBoxB']) / 2
    data['RelativeHeightPosition'] = np.where(centerY_A < centerY_B, 1, 0)
    return data

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Load the data
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Add features
train_data = add_features(train_data)
test_data = add_features(test_data)

# Print the first few rows of the train data to understand its structure after adding features
print("Data after adding features:")
print(train_data.head())

# Drop non-relevant columns
X_train = train_data.drop(['ImageID', 'Annotation', 'ImageUrl'], axis=1)
y_train = train_data['Annotation']
X_test = test_data.drop(['ImageID', 'Annotation', 'ImageUrl'], axis=1)
y_test = test_data['Annotation']

# Scale the features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Train a logistic regression model with class_weight set to 'balanced' and increased max_iter
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = clf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Display the classification report
print(classification_report(y_test, y_pred))
