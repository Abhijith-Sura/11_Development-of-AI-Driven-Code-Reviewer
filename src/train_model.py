import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create necessary folders
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION - TRAINING")
print("=" * 50)

# Step 1: Load Dataset
print("\n[Step 1] Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"✓ Dataset loaded successfully")
print(f"  - Total samples: {len(X)}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Classes: {len(target_names)} ({', '.join(target_names)})")

# Step 2: Create DataFrame and save to CSV
print("\n[Step 2] Creating DataFrame and saving to CSV...")
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.to_csv('data/iris.csv', index=False)
print(f"✓ Data saved to data/iris.csv")

# Step 3: Data Statistics
print("\n[Step 3] Dataset Statistics:")
print(df.groupby('species_name').size())

# Step 4: Split Data
print("\n[Step 4] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# Step 5: Feature Scaling
print("\n[Step 5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Step 6: Train Multiple Models
print("\n[Step 6] Training multiple models...")
print("-" * 50)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"✓ {name} - Accuracy: {accuracy * 100:.2f}%")

# Step 7: Select Best Model
print("\n[Step 7] Model Comparison:")
print("-" * 50)
for name, acc in results.items():
    print(f"{name}: {acc * 100:.2f}%")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✓ Best Model: {best_model_name} ({results[best_model_name] * 100:.2f}%)")

# Step 8: Save Model
print("\n[Step 8] Saving model and scaler...")
joblib.dump(best_model, 'models/knn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Model saved to models/knn_model.pkl")
print("✓ Scaler saved to models/scaler.pkl")

# Step 9: Final Evaluation
print("\n[Step 9] Final Model Evaluation:")
print("-" * 50)
y_pred_final = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_final, target_names=target_names))

print("\n" + "=" * 50)
print("✓ TRAINING COMPLETE!")
print("=" * 50)
