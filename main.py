from flask import Flask, render_template
import json
from flask import request, jsonify
import time
# app = Flask(__name__)
from Checking_Prediction import check_class

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=="POST":
        files = request.files
        print(files)
        if len(files):
            file = files.get("chooseFile")
            filename = file.filename
            file.save(file.filename)
            label, probability = check_class(filename)
            print(label, probability)
            return render_template("main.html", label=label.upper(), confidence=probability)
    return render_template("main.html")


@app.route('/send_file', methods=['POST'])
def send_file():
    if request.method=="POST":
        files = request.files
        print(files)
        if len(files):
            file = files.get("chooseFile")
            filename = file.filename
            file.save(file.filename)
            label, probability = check_class(filename)
            print(label, probability)
            return jsonify({"success": True, "data": {"label": label, "probability": probability}})
        else:
            return jsonify({"success": False, "message": "file missing!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
