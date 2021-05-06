import flask
import torch
import cv2
from numpy import fromfile, uint8
import torchvision.transforms as transforms
from models import model


app = flask.Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return "Welcome to the prediction app"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if flask.request.method == "POST":
        uploaded_file = flask.request.files['file']
        if uploaded_file.filename != '':
            uploaded_file = flask.request.files['file']
            uploaded_file = fromfile(uploaded_file, uint8)
            uploaded_file = cv2.imdecode(uploaded_file, cv2.IMREAD_COLOR)
            uploaded_file = augment_data_valid(uploaded_file).unsqueeze(0)
            predictor = model.ResNet18(out_activations=10, in_channels=3)
            saved_state = torch.load("/home/jedrzej/Desktop/ML_app/models/predictor.pkl")["state_dict"]
            predictor.load_state_dict(saved_state)
            output = predictor(uploaded_file)
            
    return flask.render_template("request.html")

def augment_data_valid(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_transforms = transforms.Compose([transforms.ToTensor(),
                                            normalize,])
    image = valid_transforms(image)
    return image

if __name__ == "__main__":
    app.run(debug=True)