from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import joblib as jb
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os  # For accessing environment variables

# Load the saved models  
modelmaid = jb.load("maidprice_model.pkl")
modelabj = jb.load("abjprice_model.pkl")
modellag = jb.load("lagprice_model.pkl")
modelkano = jb.load("kanoprice_model.pkl")

# Create Flask app
app = Flask(__name__)

# Define home route to display an input form
@app.route("/")
def home():
    return render_template("index.html")  # HTML file for user input

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    global df
    try:
        # Extract the area and location from the form input
        area_input = request.form["area"]
        location_input = request.form["location"]

        # Validate input type
        try:
            area = float(area_input)  # Convert input to float
        except ValueError:
            raise ValueError("Input must be a numeric value.")

        if location_input == "Abuja":
            # Prepare the data in a 2D array and make the prediction
            features_array = np.array([[area]])
            prediction = modelabj.predict(features_array)
            predicted_price = round(prediction[0], 2)
        elif location_input == "Lagos":
            # Prepare the data in a 2D array and make the prediction
            features_array = np.array([[area]])
            prediction = modellag.predict(features_array)
            predicted_price = round(prediction[0], 2)
        elif location_input == "Maiduguri":
            # Prepare the data in a 2D array and make the prediction
            features_array = np.array([[area]])
            prediction = modelmaid.predict(features_array)
            predicted_price = round(prediction[0], 2)
        elif location_input == "Kano":
            # Prepare the data in a 2D array and make the prediction
            features_array = np.array([[area]])
            prediction = modelkano.predict(features_array)
            predicted_price = round(prediction[0], 2)


        # Render the output page with the results
        return render_template("output.html", area=area, predicted_price = predicted_price)
    
    except Exception as e:
        # Render the error page for any exception
        return render_template("error.html", error_message=str(e))
    

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
