import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data
X = np.array([[30, 70], 
              [25, 80], 
              [27, 60], 
              [31, 65], 
              [23, 85], 
              [28, 75]])

y = np.array([0, 1, 0, 0, 1, 1])

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X, y)

# New data point
new_point = np.array([[26, 78]])

# Prediction
prediction = knn.predict(new_point)[0]

# Plotting
plt.figure(figsize=(7, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1], s=100, label="Sunny (0)")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=100, label="Rainy (1)")

plt.scatter(new_point[0, 0], new_point[0, 1], 
            marker='*', s=300, color='red', label="New Prediction")

plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("KNN Weather Classification")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Print result
if prediction == 0:
    print("Predicted Weather: Sunny")
else:
    print("Predicted Weather: Rainy")
  
