import joblib
import numpy as np

# Load trained model and scaler
print("Loading model...")
model = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("✓ Model loaded successfully\n")

species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict Iris species based on measurements
    """
    # Create feature array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Input Measurements:")
    print(f"  Sepal Length: {sepal_length} cm")
    print(f"  Sepal Width:  {sepal_width} cm")
    print(f"  Petal Length: {petal_length} cm")
    print(f"  Petal Width:  {petal_width} cm")
    print("-" * 60)
    print(f"Predicted Species: {species_map[prediction]}")
    print(f"Confidence: {probabilities[prediction] * 100:.2f}%")
    print("-" * 60)
    print("All Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {species_map[i]}: {prob * 100:.2f}%")
    print("=" * 60)
    print()
    
    return species_map[prediction]

if __name__ == "__main__":
    print("Testing predictions on sample flowers...\n")
    
    # Test 1: Setosa (small petals)
    print("[Test 1] Small flower (likely Setosa)")
    predict_iris(5.1, 3.5, 1.4, 0.2)
    
    # Test 2: Versicolor (medium)
    print("[Test 2] Medium flower (likely Versicolor)")
    predict_iris(6.3, 2.5, 4.9, 1.5)
    
    # Test 3: Virginica (large petals)
    print("[Test 3] Large flower (likely Virginica)")
    predict_iris(7.2, 3.6, 6.1, 2.5)
