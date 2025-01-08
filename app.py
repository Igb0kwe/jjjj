from flask import Flask, request, jsonify, render_template, send_file
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

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        area_input = request.form["area"]
        location_input = request.form["location"]

        # Process area input
        area = float(area_input)
        area = np.array([[area]])

        # Handle prediction based on location and avoid redundancy
        if location_input == "Abuja":
            prediction = modelabj.predict(area)

            global housepriceabj 

            # Add new row to the dataframe if it doesn't already exist
            if not ((housepriceabj["Area"] == area_input) & (housepriceabj["Price"] == prediction)).any():
                new_data = pd.DataFrame({"Area": [area_input], "Price": [prediction[0]]})
                housepriceabj = pd.concat([housepriceabj, new_data], ignore_index=True)

            # Plot scatter graph
            plt.scatter(housepriceabj["Area"], housepriceabj["Price"], color="blue", marker="+")
            plt.title("House Price vs Area in Abuja")
            plt.xlabel("Area (sqr ft)")
            plt.ylabel("Price (NGN)")
            plt.grid(True)
            plt.savefig("static/abj_plot.png")
            plt.close()

        elif location_input == "Lagos":
            prediction = modellag.predict(area)
            global housepricelag
            # Add new row to the dataframe if it doesn't already exist
            if not ((housepricelag["Area"] == area_input) & (housepricelag["Price"] == prediction)).any():
                new_data = pd.DataFrame({"Area": [area_input], "Price": [prediction[0]]})
                housepricelag = pd.concat([housepricelag, new_data], ignore_index=True)

            # Plot scatter graph
            plt.scatter(housepricelag["Area"], housepricelag["Price"], color="blue", marker="+")
            plt.title("House Price vs Area in Lagos")
            plt.xlabel("Area (sqr ft)")
            plt.ylabel("Price (NGN)")
            plt.grid(True)
            plt.savefig("static/lagos_plot.png")
            plt.close()

        elif location_input == "Maiduguri":
            prediction = modelmaid.predict(area)
            global housepricemaid


            # Add new row to the dataframe if it doesn't already exist
            if not ((housepricemaid["Area"] == area_input) & (housepricemaid["Price"] == prediction)).any():
                new_data = pd.DataFrame({"Area": [area_input], "Price": [prediction[0]]})
                housepricemaid = pd.concat([housepricemaid, new_data], ignore_index=True)

            # Plot scatter graph
            plt.scatter(housepricemaid["Area"], housepricemaid["Price"], color="blue", marker="+")
            plt.title("House Price vs Area in Maiduguri")
            plt.xlabel("Area (sqr ft)")
            plt.ylabel("Price (NGN)")
            plt.grid(True)
            plt.savefig("static/maiduguri_plot.png")
            plt.close()

        elif location_input == "Kano":
            prediction = modelkano.predict(area)
            global housepricekano

            # Add new row to the dataframe if it doesn't already exist
            if not ((housepricekano["Area"] == area_input) & (housepricekano["Price"] == prediction)).any():
                new_data = pd.DataFrame({"Area": [area_input], "Price": [prediction[0]]})
                housepricekano = pd.concat([housepricekano, new_data], ignore_index=True)

            # Plot scatter graph
            plt.scatter(housepricekano["Area"], housepricekano["Price"], color="blue", marker="+")
            plt.title("House Price vs Area in Kano")
            plt.xlabel("Area (sqr ft)")
            plt.ylabel("Price (NGN)")
            plt.grid(True)
            plt.savefig("static/kano_plot.png")
            plt.close()

        else:
            raise ValueError("Invalid location selected.")
        
        # Process output data
        predicted_price = int(round(prediction[0], 0))
        predicted_price = f"{predicted_price:,}"
        
        area = float(area[0][0])

        # Render the output page with the results
        return render_template("output.html", area=area, predicted_price=predicted_price, location_input=location_input)

    except Exception as e:
        # Render the error page for any exception
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
