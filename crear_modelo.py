# crear_modelo.py
from sklearn.linear_model import LogisticRegression
import joblib

# Datos de ejemplo
X = [[0, 0], [1, 1]]
y = [0, 1]

# Entrenar modelo
modelo = LogisticRegression()
modelo.fit(X, y)

# Guardar modelo
joblib.dump(modelo, 'modelo.pkl')
