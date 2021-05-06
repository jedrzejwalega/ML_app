import flask
import torch
import cv2
from numpy import fromfile, uint8
import torchvision.transforms as transforms
from models import model
import os
from werkzeug.utils import secure_filename


app = flask.Flask(__name__)
app.secret_key = "`5]-0i/M54'&{2y"
app.config['UPLOAD_FOLDER'] = "/home/jedrzej/Desktop/ML_app/static/uploads"

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
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            uploaded_file = fromfile(os.path.join(app.config['UPLOAD_FOLDER'], filename), uint8)
            uploaded_file = cv2.imdecode(uploaded_file, cv2.IMREAD_COLOR)
            image_to_predict = augment_data(uploaded_file).unsqueeze(0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predictor = model.ResNet18(out_activations=10, in_channels=3)
            saved_state = torch.load("/home/jedrzej/Desktop/ML_app/models/predictor.pkl")["state_dict"]
            predictor.load_state_dict(saved_state)
            output = predictor(image_to_predict)
            output = torch.argmax(output).item()

            classes = {
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"
            }
            
            predicted_class = classes[output]
            flask.flash(f'Image successfully uploaded and displayed below. It seems to be a {predicted_class}')
            return flask.render_template("request.html", filename=filename)

            
    return flask.render_template("request.html")

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return flask.redirect(flask.url_for('static', filename='uploads/' + filename), code=301)

def augment_data(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_transforms = transforms.Compose([transforms.ToTensor(),
                                            normalize,])
    image = valid_transforms(image)
    return image

if __name__ == "__main__":
    app.run(debug=True)