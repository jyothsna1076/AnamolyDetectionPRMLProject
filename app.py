from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import time
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Shared prediction status
prediction_status = {
    'status': 'idle',        # idle, processing, done
    'predictions': None
}

def handle_capture_and_predict():
    try:
        print("[*] Running capture_script.py (requires sudo)...")
        subprocess.run(['sudo', 'python3', 'capture_script.py'], check=True)

        print("[*] Capture complete. Running prediction...")
        subprocess.run(['python3', 'predict1.py'], check=True)

        print("[✓] Prediction complete. Reading predictions...")
        with open('/Users/pradeepikanori/PRML_project/predictions.csv', 'r') as f:
            lines = f.readlines()[1:]
            predictions = [line.strip() for line in lines]

        prediction_status['predictions'] = predictions
        prediction_status['status'] = 'done'

    except subprocess.CalledProcessError as e:
        print(f"[✗] Error during execution: {e}")
        prediction_status['status'] = 'error'
        prediction_status['predictions'] = [str(e)]
    
    except Exception as e:
        print(f"[✗] General error: {e}")
        prediction_status['status'] = 'error'
        prediction_status['predictions'] = [str(e)]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    if prediction_status['status'] == 'processing':
        return jsonify({'status': 'Already processing...'})

    prediction_status['status'] = 'processing'  # Reset here
    prediction_status['predictions'] = None     # Clear old predictions

    threading.Thread(target=handle_capture_and_predict).start()
    return jsonify({'status': 'Monitoring started...'})


@app.route('/get-predictions', methods=['GET'])
def get_predictions():
    status = prediction_status['status']
    
    if status == 'done':
        result = prediction_status['predictions']
        # Reset after serving
        prediction_status['status'] = 'idle'
        prediction_status['predictions'] = None
        return jsonify({'predictions': result})
    elif status == 'processing':
        return jsonify({'status': 'processing'})
    elif status == 'error':
        return jsonify({'error': prediction_status['predictions']})
    else:
        return jsonify({'status': 'idle', 'message': 'Click capture to start monitoring.'})

    
@app.route('/manual-check', methods=['POST'])
def manual_check():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})

        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)

        print(f"[*] Received file: {file_path}")

        # Modify this part based on your actual processing logic
        # Assuming predict1.py can take a file argument for manual check:
        result = subprocess.run(['python3', 'predict1.py', '--input', file_path], check=True)

        # Read prediction result from predictions.csv as before
        with open('/Users/pradeepikanori/PRML_project/predictions.csv', 'r') as f:
            lines = f.readlines()[1:]
            predictions = [line.strip() for line in lines]

        return jsonify({'predictions': predictions})

    except subprocess.CalledProcessError as e:
        print(f"[✗] Prediction error: {e}")
        return jsonify({'error': str(e)})
    except Exception as e:
        print(f"[✗] General error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
