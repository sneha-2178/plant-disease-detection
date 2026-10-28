import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained plant disease model
MODEL_PATH = 'plant_disease_recog_model_pwp.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# List of target class names matching your training dataset order
CLASS_NAMES = [
    'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

def model_predict(img_path, model):
    # Load image and force 160x160 size and 3-channel RGB format
    img = image.load_img(img_path, target_size=(160, 160))
    img = img.convert('RGB')
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
   
    
    preds = model.predict(x)
    print("----------------------------------")
    print("Raw Model Predictions (preds):", preds)
    print("chosen class index:", np.argmax(preds[0]))
    print("----------------------------------")

    predicted_class_index = np.argmax(preds[0])
    
    if predicted_class_index < len(CLASS_NAMES):
        return CLASS_NAMES[predicted_class_index]
    else:
        return f"Class Index: {predicted_class_index}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file uploaded.')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction='No file selected.')
    
    if file:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create a unique filename using a timestamp to avoid overwriting or caching issues
        filename = secure_filename(file.filename)
        unique_filename = str(int(time.time())) + "_" + filename
        file_path = os.path.join(upload_dir, unique_filename)
        
        file.save(file_path)
        
        # Predict the disease
        prediction = model_predict(file_path, model)
        
        # Clean up the temporary image file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)