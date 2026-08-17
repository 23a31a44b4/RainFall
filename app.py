from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Get the absolute path of this project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained ML model safely
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as file:
    ml_model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""

    if request.method == "POST":

        try:
            # Get input values
            temperature = float(request.form["temperature_celsius"])
            humidity = float(request.form["humidity"])
            pressure = float(request.form["pressure_mb"])
            wind = float(request.form["wind_kph"])
            cloud = float(request.form["cloud"])
            uv = float(request.form["uv_index"])
            visibility = float(request.form["visibility_km"])
            gust = float(request.form["gust_kph"])

            # Create feature array
            features = np.array([[
                temperature,
                humidity,
                pressure,
                wind,
                cloud,
                uv,
                visibility,
                gust
            ]])

            # Predict rainfall
            ml_prediction = ml_model.predict(features)[0]

            # Display result
            if ml_prediction == 1:
                prediction_text = "🌧️ Rainfall Expected"
            else:
                prediction_text = "☀️ No Rainfall Expected"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)