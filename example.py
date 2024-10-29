from flask import Flask, request, render_template, jsonify
from waitress import serve
from googletrans import Translator
from ultralytics import YOLO
from flask import Response
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("check.html")

@app.route("/detect", methods=["POST"])
def detect():

    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),
      mimetype='application/json'
    )


def detect_objects_on_image(buf):


    model = YOLO("yolov8m.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    label = data['data1']
    language = data['data2']
    # Translate the label to French
    translator = Translator()
    translation = translator.translate(label, dest=language)
    translated_text = translation.text

    # Send the translated text back to HTML
    return jsonify({'translatedLabel': translated_text})

if __name__ == "__main__":
    # Use '0.0.0.0' to make it accessible from outside the local machine
    serve(app, host='0.0.0.0', port=8080)
