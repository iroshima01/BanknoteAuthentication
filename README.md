# BanknoteAuthentication
# BBM409 Assignment 1 - Fall 2024

## Perceptron Learning Algorithm Implementation

This project involves implementing the **Perceptron Learning Algorithm** to classify the **Banknote Authentication Dataset**. The dataset contains four features: **variance, skewness, kurtosis, and entropy**, with a binary classification target (authentic: `1`, fake: `0`).

---

## **Step 1: Data Preparation**

### **1. Download and Load the Data**
- Download the **Banknote Authentication Dataset** from the UCI repository.
- Load the dataset in Python using Pandas:
  
  ```python
  import pandas as pd
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
  columns = ["variance", "skewness", "kurtosis", "entropy", "class"]
  df = pd.read_csv(url, names=columns)
  ```

### **2. Preprocess the Data**
- Analyze whether preprocessing is needed (e.g., **normalization**).
- Since Perceptron is sensitive to large feature scales, normalization may improve convergence.

  ```python
  from sklearn.preprocessing import StandardScaler
  
  X = df.iloc[:, :-1].values  # Features
  y = df.iloc[:, -1].values   # Target labels
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

### **3. Training and Validation Split**
- Split the dataset into **80% training** and **20% validation**:
  
  ```python
  from sklearn.model_selection import train_test_split
  
  X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
  ```

- This split ensures that the model generalizes well to unseen data instead of memorizing training data.

---

## **Step 2: Initialize the Perceptron**

### **1. Initialization**
- Initialize the **weight vector (w)** and **bias (b)** to small random values or zeros.
- Set hyperparameters: **learning rate (η)** and **number of epochs**.
  
  ```python
  import numpy as np
  
  np.random.seed(42)
  w = np.random.randn(X_train.shape[1]) * 0.01  # Small random values
  b = 0  # Bias initialized to zero
  learning_rate = 0.01
  epochs = 1000
  ```

### **2. Effect of Learning Rate and Epochs**
- **Too small η** → Slow convergence, may get stuck.
- **Too large η** → Overshooting, may never converge.
- **More epochs** → Better training but risks overfitting.

---

## **Step 3: Train the Perceptron**

### **1. Training Loop**
- Iterate through **each sample** in the training set.
- Compute **weighted sum**: 
  
  ```python
  def step_function(z):
      return 1 if z >= 0 else 0
  
  for epoch in range(epochs):
      for i in range(len(X_train)):
          z = np.dot(w, X_train[i]) + b
          y_pred = step_function(z)
          
          # Weight update rule
          w += learning_rate * (y_train[i] - y_pred) * X_train[i]
          b += learning_rate * (y_train[i] - y_pred)
  ```

### **2. Handling Linearly Separable vs. Non-Linearly Separable Data**
- **Linearly separable**: Perceptron converges.
- **Non-linearly separable**: Perceptron **does not converge**, as it only works for **linear decision boundaries**.

### **3. Model Evaluation**
- Apply trained Perceptron on training data and compute predictions:

  ```python
  y_train_pred = [step_function(np.dot(w, x) + b) for x in X_train]
  y_val_pred = [step_function(np.dot(w, x) + b) for x in X_val]
  ```

- Evaluate model using accuracy:

  ```python
  from sklearn.metrics import accuracy_score
  
  train_acc = accuracy_score(y_train, y_train_pred)
  val_acc = accuracy_score(y_val, y_val_pred)
  
  print(f"Training Accuracy: {train_acc * 100:.2f}%")
  print(f"Validation Accuracy: {val_acc * 100:.2f}%")
  ```

---
