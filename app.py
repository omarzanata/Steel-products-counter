from flask import Flask, render_template, request, jsonify
from IPython.display import Image, display
from ultralytics import YOLO
import numpy as np
import glob

app = Flask(__name__,template_folder='templates')

# Load YOLO model
model_path = 'C:/Users/KimoStore/Downloads/runs/detect/train15/weights/best.pt'
model = YOLO(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
     if request.method == 'POST':
         # Check if a file was uploaded
         if 'file' not in request.files:
             return jsonify({"error": "No file provided"}), 400

         file = request.files['file']

         # Check if the file has an allowed extension
         if file.filename == '':
             return jsonify({"error": "No selected file"}), 400

         # Save the uploaded file
         file_path = "static/Uploads/" + file.filename
         file.save(file_path)

         # Perform object detection
         result = model.predict(source=file_path, conf=0.45, save=True)

         #file_path2 = "runs/detect/predict" + file.filename
         #file.save(file_path2)
         #display(Image(filename=file_path2, width=600))
         # Display the result image
         display(Image(filename=file_path, width=600))

         # Get coordinates
         arrxy = result[0].boxes.xyxy
         coordinates = np.array(arrxy)
         x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2
         y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2
         midpoints = np.column_stack((x_coords, y_coords))
         rounded_n_sorted_arr = np.round(midpoints[midpoints[:, 1].argsort()]).astype(int)

         # Prepare results for rendering
         count_message = "Number of coils detected: " + str(len(rounded_n_sorted_arr))

         return render_template('index.html', count_message=count_message,filename=file.filename)

     return render_template('index.html')

if __name__ == '__main__':

    app.run(debug=True)
