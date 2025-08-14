import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\\Users\\duhdu\\OneDrive\\Desktop\\data.csv")  
z = data.iloc[:, 0].values 
y = data.iloc[:, 1].values  
x = data.iloc[:, 2].values  

X = np.column_stack((x, y, x**2, y**2, x*y, np.ones_like(x)))
feature_names = ["x", "y", "x^2", "y^2", "x*y"]
model = LinearRegression()
model.fit(X, z)

np.set_printoptions(suppress=True)
print("Regression Parameters:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.9f}")
print(f"Intercept term: {model.intercept_:.9f}")
r2 = model.score(X, z)
print(f"R^2: {r2:.6f}")

x_range = np.linspace(x.min(), x.max(), 30)
y_range = np.linspace(y.min(), y.max(), 30)
x_grid, y_grid = np.meshgrid(x_range, y_range)
X_pred = np.column_stack((x_grid.ravel(), y_grid.ravel(),
                          x_grid.ravel()**2, y_grid.ravel()**2,
                          x_grid.ravel()*y_grid.ravel(),
                          np.ones_like(x_grid.ravel())))
z_pred = model.predict(X_pred).reshape(x_grid.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='red', label='Data Points')
ax.plot_surface(x_grid, y_grid, z_pred, alpha=0.5, cmap='viridis')
ax.set_xlabel('length')
ax.set_ylabel('mass')
ax.set_zlabel('clamp')
plt.legend()
plt.show()
