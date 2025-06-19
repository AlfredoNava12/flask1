from flask import Flask
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo.pkl')

@app.route('/')
def inicio():
    return "¡Clasificador activo! Visita /predecir para ver una predicción."

@app.route('/predecir')
def predecir():
    # Usamos un ejemplo de entrada [0, 0]
    resultado = modelo.predict([[0, 0]])
    return f"La predicción del modelo para [0, 0] es: {resultado[0]}"

if __name__ == '__main__':
    app.run()
