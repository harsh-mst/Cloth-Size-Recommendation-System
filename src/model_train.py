import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('improved_size_balanced_empty.csv')
X = df[['weight', 'height']]
y = df['size']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
model.fit(X_train, y_train)


with open('size_recommender_model_v4.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model retrained and saved with current scikit-learn version!")
