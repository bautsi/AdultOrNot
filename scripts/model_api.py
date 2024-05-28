from flask import Flask, request, jsonify
from flask_cors import CORS  # 導入 CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # 啟用 CORS
model = load_model('models\specific_self_model_3.keras')  # 確保使用正確的模型路徑

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part")
        return jsonify(error="No file uploaded"), 400
    
    img_file = request.files['file']
    if img_file:
        filename = 'temp_image.jpg'
        img_path = os.path.join('data/temp_images', filename)
        img_file.save(img_path)
        print(f"Image saved to {img_path}")

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        os.remove(img_path)
        print(f"Image removed {img_path}")

        result = "Adult" if prediction[0][0] > 0.5 else "Minor"
        response = jsonify(result=result)
        print("Prediction:", prediction)
        print("Result sent:", result)
        print("JSON response:", response.get_json())  # 確認 JSON 響應的內容
        return response

    return jsonify(error="No file uploaded"), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
