import os
from flask import Flask, request, jsonify
from etl_pipeline import DataPipeline
from werkzeug.utils import secure_filename

# make the Flask app to serve the model as a microservice
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'  # specify the folder where the uploaded videos will be saved
ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi'}  # specify the allowed video file extensions

@app.route('/predict', methods=['POST'])
def predict_license_plates():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        video_path = os.path.join(UPLOAD_FOLDER, filename)

        # Start the ETL pipeline
        dp = DataPipeline()
        dp.extract(video_path)
        dp.transform()
        predictions = dp.load()
        return jsonify({'predictions': predictions})
    else:
        return jsonify({'error': 'Invalid file format'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    flaskPort = 8000  # This is the port number you want to use
    print('starting server...')
    app.run(host='0.0.0.0', port=flaskPort)