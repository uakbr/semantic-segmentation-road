import os
import tempfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from model.inference import inference_video
from visualization.visualize import visualize_video

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file uploaded', 400
    
    video = request.files['video']
    if video.filename == '':
        return 'No video selected', 400
    
    filename = secure_filename(video.filename)
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, filename)
    video.save(video_path)
    
    segmentation_maps = inference_video(video_path)
    
    output_path = os.path.join(temp_dir, 'output.mp4')
    visualize_video(video_path, segmentation_maps, output_path)
    
    return send_file(output_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)