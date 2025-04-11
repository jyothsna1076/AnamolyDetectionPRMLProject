from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import time
import os

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
            lines = f.readlines()[1:]  # skip header
            predictions = [line.strip() for line in lines]

        prediction_status['predictions'] = predictions
        prediction_status['status'] = 'done'

    except subprocess.CalledProcessError as e:
        print(f"[✗] Error during execution: {e}")
        prediction_status['status'] = 'error'
        prediction_status['predictions'] = [str(e)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    if prediction_status['status'] == 'processing':
        return jsonify({'status': 'Already processing...'})
    
    threading.Thread(target=handle_capture_and_predict).start()
    return jsonify({'status': 'Monitoring started...'})

@app.route('/get-predictions', methods=['GET'])
def get_predictions():
    status = prediction_status['status']
    
    if status == 'done':
        return jsonify({'predictions': prediction_status['predictions']})
    elif status == 'processing':
        return jsonify({'status': 'processing'})
    elif status == 'error':
        return jsonify({'error': prediction_status['predictions']})
    else:
        return jsonify({'status': 'idle', 'message': 'Click capture to start monitoring.'})

if __name__ == '__main__':
    app.run(debug=True)
