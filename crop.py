import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load data
data = pd.read_csv("Crop_recommendation.csv")

# Encoding
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
data['crop_num'] = data['label'].map(crop_dict)
data.drop(['label'], axis=1, inplace=True)

# Train test split
X = data.drop(['crop_num'], axis=1)
y = data['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

# Model training
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print(f"{name} with accuracy: {accuracy_score(y_test, ypred)}")
    print("Confusion matrix:", confusion_matrix(y_test, ypred))
    print("==========================================================")

# Final model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
ypred = rfc.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, ypred))

# Save model and scaler
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))

# Recommendation function
def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    prediction = rfc.predict(transformed_features)
    print(prediction)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    N = 40
    P = 50
    k = 50
    temperature = 40.0
    humidity = 20
    ph = 100
    rainfall = 100
    predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)
    crop_dict_rev = {v: k.capitalize() for k, v in crop_dict.items()}
    if predict in crop_dict_rev:
        print(f"{crop_dict_rev[predict]} is a best crop to be cultivated.")
    else:
        print("Sorry, not able to recommend a proper crop for this environment.")
