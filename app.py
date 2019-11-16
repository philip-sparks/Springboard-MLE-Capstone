"""
This file contains the codes that productionalize the classifier
"""

# standard imports
import json

# external imports
from flask import Flask, request

# local imports
from stored_model_predictor import StoredModelPredictor

app = Flask(__name__)


@app.route("/")
def hello_world():
    """
    Display an initial message when user goes to the IP address
    """
    return """
        Welcome!

        The email classifier is running on this address.

        Supply a json containing the 'Body' key to the /predict endpoint.

        Copy/paste the following curl command, changing the body of the email and the current EC2 instance:

        curl -H "Content-Type: application/json" -d '{"Body" : "starting from July 1 you have the obligation to file claims for new information"}' http://ec2-18-217-36-44.us-east-2.compute.amazonaws.com:1234/predict
        """


@app.route("/predict", methods=["POST"])
def predict():
    """
    Process a post request supplying a JSON payload.
    """
    if request.method == "POST":
        prediction = stored_predictor_instance.predict(request.json)
        json_str = json.dumps(prediction)
        return str(json_str)


@app.before_first_request
def load_model():
    """
    Read the model from S3 and prepare the classifier
    """
    global stored_predictor_instance
    stored_predictor_instance = StoredModelPredictor()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234)
