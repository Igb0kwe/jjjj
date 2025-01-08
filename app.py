from flask import Flask, request, render_template
import numpy as np
import joblib as jb
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the saved models
modelmaid = jb.load("maidprice_model.pkl")
modelabj = jb.load("abjprice_model.pkl")
modellag = jb.load("lagprice_model.pkl")
modelkano = jb.load("kanoprice_model.pkl")

# Load dataframes
housepriceabj = pd.read_csv("homepricesabj.csv")
housepricelag = pd.read_csv("homepriceslag.csv")
housepricemaid = pd.read_csv("homepricesmaid.csv")
housepricekano = pd.read_csv("homepriceskano.csv")

# Create Flask app
app = Flask(__name__)

# Define home route to display an input form
@app.route("/")
def home():
    return render_template("index.html")  # HTML file for user input

def handle_prediction(location_input, area_input, model, house_df, plot_filename):
    area = float(area_input)
    area_array = np.array([[area]])
    
    # Predict the price using the selected model
    prediction = model.predict(area_array)
    
    # Add new row to the dataframe if it doesn't already exist
    if not ((house_df["Area"] == area_input) & (house_df["Price"] == prediction[0])).any():
        new_data = pd.DataFrame({"Area": [area_input], "Price": [prediction[0]]})
        house_df = pd.concat([house_df, new_data], ignore_index=True)
    
    # Plot scatter graph
    plt.scatter(house_df["Area"], house_df["Price"], color="blue", marker="+")
    plt.title(f"House Price vs Area in {location_input}")
    plt.xlabel("Area (sqr ft)")
    plt.ylabel("Price (NGN)")
    plt.grid(True)
    plt.savefig(f"static/{plot_filename}")
    plt.close()
    
    return prediction, house_df

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Declare the global variables
    global housepriceabj, housepricelag, housepricemaid, housepricekano  
    
    try:
        area_input = request.form["area"]
        location_input = request.form["location"]

        if location_input == "Abuja":
            prediction, housepriceabj = handle_prediction("Abuja", area_input, modelabj, housepriceabj, "abj_plot.png")
        elif location_input == "Lagos":
            prediction, housepricelag = handle_prediction("Lagos", area_input, modellag, housepricelag, "lagos_plot.png")
        elif location_input == "Maiduguri":
            prediction, housepricemaid = handle_prediction("Maiduguri", area_input, modelmaid, housepricemaid, "maiduguri_plot.png")
        elif location_input == "Kano":
            prediction, housepricekano = handle_prediction("Kano", area_input, modelkano, housepricekano, "kano_plot.png")
        else:
            raise ValueError("Invalid location selected.")
        
        # Process output data
        predicted_price = int(round(prediction[0], 0))
        predicted_price = f"{predicted_price:,}"
        area = float(area_input)

        # Render the output page with the results
        return render_template("output.html", area=area, predicted_price=predicted_price, location_input=location_input)

    except Exception as e:
        # Render the error page for any exception
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)