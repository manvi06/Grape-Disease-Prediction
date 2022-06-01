from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

labels = np.array(['Black Rot', 'Grape Esca', 'Grape Healthy', 'Leaf Blight'])

model = load_model('model.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path)
    i = image.img_to_array(i)/255.0
    npp_image = np.expand_dims(i, axis=0)
    p = model.predict(npp_image)
    itemindex = np.where(p == np.max(p))
    return labels[itemindex[1][0]]


# routes
@ app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@ app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
