from flask import Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
import requests
import json
from test import trained_model
app = Flask(__name__)
ERR_STATUS, OK_STATUS = 400, 200

model = trained_model()

@app.route('/test/<string:path>', methods=['GET'])
def get_tested_image_result(path):
    return str(model.test_one(f'test/{path}')), OK_STATUS


@app.route('/test', methods=['POST'])
def post_test_image():
    #json_string = json.dumps(request.get_json(), indent=4)
    if 'file' not in request.files:
        flash('No file part')
        return "", ERR_STATUS
    file = request.files['file']
    #if file.filename == '':
    #    flash('No image selecterd for uploading')
    #    return "", ERR_STATUS
    
    #file_name = secure_filename(file.filename)
    #file.save(file_name)
    result = model.test_one_image(file.stream)

    return str(result), OK_STATUS



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)