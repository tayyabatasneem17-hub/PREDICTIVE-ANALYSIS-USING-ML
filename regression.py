# TASK-2: Predictive Analysis using Machine Learning (Regression Example)
# Internship - CODTECH

# ---------------------------
# Step 1: Import libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Step 2: Create sample dataset (Years of Experience vs Salary)
# ---------------------------
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0,
                        3.2, 3.2, 3.7, 3.9, 4.0, 4.0,
                        4.1, 4.5, 4.9, 5.1, 5.3, 5.9,
                        6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 10.3],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150,
               54445, 64445, 57189, 63218, 55794, 56957,
               57081, 61111, 67938, 66029, 83088, 81363,
               93940, 91738, 98273, 101302, 113812, 109431,
               105582, 116969, 112635]
}
df = pd.DataFrame(data)

print("===== Dataset Sample =====")
print(df.head())

# ---------------------------
# Step 3: Feature Selection
# ---------------------------
X = df[['YearsExperience']]  # Independent variable
y = df['Salary']             # Dependent variable

# ---------------------------
# Step 4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 5: Model Training
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Step 6: Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Step 7: Model Evaluation
# ---------------------------
print("\n===== Model Evaluation =====")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------
# Step 8: Visualization
# ---------------------------
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.legend()
plt.show()