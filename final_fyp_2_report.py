# -*- coding: utf-8 -*-

# --- Imports ---
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
# --- End Imports ---

# --- Data Loading ---
df = pd.read_csv(r'F:\FYP\APP\Pakistan_Major_Crops_High_Favourability_Dataset_With_Moisture.csv')
print(df)
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.columns)
# --- End Data Loading ---

# --- Feature Selection ---
numerical_features = ['Temperature', 'N', 'P', 'K', 'pH', 'EC', 'Moisture']
print(numerical_features)
print(df[df['Crop'] == 'Wheat'])
categorical_features = ['Crop']
print(categorical_features)
print(type(categorical_features))
print(type(numerical_features))
# --- End Feature Selection ---

# --- Label Encoding ---
labels = list(df['Crop'].unique())
print(labels)
le_data = df.copy()
label_encoder = preprocessing.LabelEncoder()
le_data['Crop'] = label_encoder.fit_transform(le_data['Crop'])
print(le_data)
# --- End Label Encoding ---

# --- Boxplots for Numerical Features ---
for col in numerical_features:
    le_data[col] = pd.to_numeric(le_data[col], errors='coerce')
    le_data = le_data.dropna(subset=[col])
    le_data.boxplot(column=col, by='Crop', figsize=(7, 4))
    plt.title("")
    plt.xlabel(col)
plt.show()
# --- End Boxplots ---

# --- Data Description ---
print(le_data.shape)
print(le_data.describe())
# --- End Data Description ---

# --- Correlation Heatmap ---
correlation = le_data.corr()
plt.figure(figsize=(12,5), dpi=600)
sns.heatmap(correlation, annot=True)
plt.show()
# --- End Correlation Heatmap ---

# --- Crop Distribution Pie Chart ---
labels = list(df.Crop.unique())
sizes = []
for i in range(len(labels)):
    sizes.append((df.Crop.value_counts())[i])
print(df.Crop.value_counts())
plt.figure(figsize=(13,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.legend(labels)
plt.show()
# --- End Pie Chart ---

# --- Histograms for Numerical Features ---
for column in numerical_features:
    plt.figure(figsize=(20,4))
    sns.histplot(data=le_data, x=column, bins=100, kde=True, hue='Crop', palette='husl')
    plt.title(column)
# --- End Histograms ---

# --- Train-Test Split ---
from sklearn.model_selection import train_test_split
X = le_data[numerical_features]
y = le_data['Crop']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# --- End Train-Test Split ---

# --- Logistic Regression ---
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
print(lr_model)
from sklearn.metrics import classification_report, confusion_matrix
prediction_lr = lr_model.predict(X_test)
print('Predicted labels: ', np.round(prediction_lr)[:10])
print('Actual labels   : ', y_test[:10])
print(classification_report(y_test, prediction_lr))
cm = confusion_matrix(y_test, prediction_lr)
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')
plt.show()
# --- End Logistic Regression ---

# --- Naive Bayes ---
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
prediction_nb = gnb_model.predict(X_test)
print('Predicted labels: ', np.round(prediction_nb)[:10])
print('Actual labels   : ', y_test[:10])
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, prediction_nb))
print(classification_report(y_test, prediction_nb))
cm = confusion_matrix(y_test, prediction_nb)
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')
plt.show()
# --- End Naive Bayes ---

# --- Random Forest ---
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
prediction_rf = rf_model.predict(X_test)
print('Predicted labels: ', np.round(prediction_rf)[:10])
print('Actual labels   : ', y_test[:10])
print('Accuracy: ', accuracy_score(y_test, prediction_rf))
print(classification_report(y_test, prediction_rf))
cm = confusion_matrix(y_test, prediction_rf)
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')
plt.show()
# --- End Random Forest ---

# --- k-Nearest Neighbour ---
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
prediction_knn = knn_model.predict(X_test)
print('Predicted labels: ', np.round(prediction_knn)[:10])
print('Actual labels   : ', y_test[:10])
print('Accuracy: ', accuracy_score(y_test, prediction_knn))
print(classification_report(y_test, prediction_knn))
cm = confusion_matrix(y_test, prediction_knn)
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')
plt.show()
# --- End kNN ---

# --- SVM ---
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# --- End SVM ---

# --- Model Comparison ---
models = pd.DataFrame({'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'k-Nearest Neighbour', 'SVM'],
                      'Score': [accuracy_score(y_test, prediction_lr), accuracy_score(y_test, prediction_nb), accuracy_score(y_test, prediction_rf), accuracy_score(y_test, prediction_knn), accuracy_score(y_test, y_pred)]})
print(models.sort_values(by='Score', ascending=False))
plt.figure(figsize=(10,5))
sns.barplot(x='Model', y='Score', data=models)
plt.show()
# --- End Model Comparison ---

# --- Data Inspection ---
print("Unique crops and counts:")
print(le_data['Crop'].value_counts())
print("Train target distribution:")
print(y_train.value_counts())
print("Training feature summary:")
print(X_train.describe())
# --- End Data Inspection ---

# --- User Input Prediction ---
le = preprocessing.LabelEncoder()
n = float(input("Enter nitrogen level (N): "))
p = float(input("Enter phosphorus level (P): "))
k = float(input("Enter potassium level (K): "))
temperature = float(input("Enter temperature (°C): "))
ph = float(input("Enter pH level: "))
ec = float(input("Enter EC level: "))
moisture = float(input("Enter moisture level: "))
user_input = {
    'Temperature': temperature,
    'N': n,
    'P': p,
    'K': k,
    'pH': ph,
    'EC': ec,
    'Moisture': moisture
}
print("\nCollected Sensor Data:")
print(user_input)
le.fit(le_data['Crop'])
user_df = pd.DataFrame([user_input])
user_df = user_df[numerical_features]
prediction = gnb_model.predict(user_df)[0]
predicted_crop = le.inverse_transform([prediction])[0]
print("🌾 Recommended Crop:", predicted_crop)
# --- End User Input Prediction ---

# --- Save Model ---
import joblib
joblib.dump(gnb_model, 'crop_prediction_model.pkl')

# --- End Save Model ---

# --- SVM Sample Predictions ---
for i in range(5):
    sample = X_test.iloc[i].to_frame().T
    pred = svm_model.predict(sample)[0]
    print(f"Prediction {i+1}: {pred}")
# --- End SVM Sample Predictions ---

