# Model Research for Iris Flower Classification

## Problem Statement
Classify Iris flowers into 3 species (Setosa, Versicolor, Virginica) based on 4 measurements:
- Sepal Length
- Sepal Width  
- Petal Length
- Petal Width

## Dataset
- **Name**: Iris Dataset
- **Samples**: 150 (50 per species)
- **Features**: 4 numerical features
- **Target**: 3 classes (multi-class classification)

## Models Researched

### 1. Logistic Regression
**Description**: Linear model for classification that uses logistic function.

**Advantages**:
- Simple and fast to train
- Works well for linearly separable data
- Easy to interpret

**Disadvantages**:
- Assumes linear relationship between features
- May underperform on complex patterns

**Expected Accuracy**: 95-97%

---

### 2. K-Nearest Neighbors (KNN)
**Description**: Instance-based learning that classifies based on k nearest data points.

**Advantages**:
- No training phase required
- Works well with small datasets
- Can capture non-linear patterns

**Disadvantages**:
- Slow prediction on large datasets
- Sensitive to feature scaling

**Expected Accuracy**: 97-100%

---

### 3. Decision Tree
**Description**: Tree-based model that splits data based on feature values.

**Advantages**:
- Easy to visualize and understand
- Handles non-linear relationships
- No feature scaling needed

**Disadvantages**:
- Can easily overfit
- Unstable with small changes in data

**Expected Accuracy**: 95-98%

---

## Model Selection
**Selected Model**: K-Nearest Neighbors (k=5)

**Justification**:
- Best accuracy on Iris dataset (historically 97-100%)
- Works well with small, clean datasets
- Simple to implement and understand
- No training time required

## References
- Scikit-learn Documentation
- UCI Machine Learning Repository - Iris Dataset
- "Pattern Recognition and Machine Learning" by Christopher Bishop
