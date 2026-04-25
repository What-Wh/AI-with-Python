import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv('./assets/internship_candidates_final_numeric.csv')

X = df[['Experience','Grade','EnglishLevel','Age','EntryTestScore']]
Y = df['Accepted']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_prec = model.predict(X_test)

plt.scatter(X_test['Experience'], X_test['EntryTestScore'], c=Y_prec, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('Experience')
plt.ylabel('Entry Test Score')
plt.colorbar(label='Predicted Class')
plt.show()

print("Accuracy:", accuracy_score(Y_test, Y_prec))
print("Precision:", precision_score(Y_test, Y_prec))
print("Recall:", recall_score(Y_test, Y_prec))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prec))