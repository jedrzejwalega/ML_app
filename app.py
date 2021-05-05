import flask
import torch

app = flask.Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return "Welcome to the prediction app"

@app.route("/predict")
def predict():
    return "Here you can predict whether your picture is cat or dog"

if __name__ == "__main__":
    app.run(debug=True)