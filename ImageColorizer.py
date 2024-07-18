from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from cv2 import dnn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['DOWNLOAD_FOLDER'] = './downloads/'

# Create upload and download directories if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #--------Model file paths--------#
        proto_file = 'model\colorization_deploy_v2.prototxt'
        model_file = 'model\colorization_release_v2.caffemodel'
        hull_pts = 'model\pts_in_hull.npy'
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        #--------Reading the model params--------#
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)

        #-----Reading and preprocessing image--------#
        img = cv2.imread(img_path)
        scaled = img.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        # we'll resize the image for the network
        resized = cv2.resize(lab_img, (224, 224))
        # split the L channel
        L = cv2.split(resized)[0]
        # mean subtraction
        L -= 50

        # predicting the ab channels from the input L channel
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        # resize the predicted 'ab' volume to the same dimensions as our input image
        ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

        # Take the L channel from the image
        L = cv2.split(lab_img)[0]
        # Join the L channel with predicted ab channel
        colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

        # Then convert the image from Lab to BGR 
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # change the image to 0-255 range and convert it from float32 to int
        colorized = (255 * colorized).astype("uint8")
        colorized_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        cv2.imwrite(colorized_path, colorized)

        return redirect(url_for('display_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/downloads/<filename>')
def colorized_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

@app.route('/download_colorized/<filename>', methods=['GET'])
def download_colorized(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/display/<filename>')
def display_file(filename):
    return render_template('disp.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
