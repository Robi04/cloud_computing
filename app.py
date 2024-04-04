from flask import Flask, request, Response, send_file, jsonify
from flask import render_template
from PIL import Image
import base64
from io import BytesIO
from main import *
# import main as RCNN



app = Flask(__name__)


@app.route("/")
def index(name=None):
    return render_template("index.html", name=name)


@app.route("/segment", methods=["POST"])
def segment():
    file = request.files["image"]
    file_content = file.read()
    img = Image.open(BytesIO(file_content))
    data = base64.b64encode(file_content).decode()
    return f'<img style="width:600px" src="data:image/png;base64,{data}">'
