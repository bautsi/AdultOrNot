from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # 啟用CORS，允許跨域請求

model_path = 'models\VGG16_range_2_model.keras'
model = load_model(model_path)  # 載入模型
print(f"Model loaded from {model_path}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    if file:
        filename = 'temp_image.jpg'
        img_path = os.path.join('data/temp_images', filename)
        file.save(img_path)
        print(f"Image saved to {img_path}")

        # 圖片處理和預測
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # 正規化
        prediction = model.predict(img_array)[0][0]
        
        os.remove(img_path)  # 刪除暫存圖片
        print(f"Image removed from {img_path}")

        # 根據您的模型輸出適當修改這部分
        result = f"Prediction: {prediction}"
        return jsonify(result=result)

    return jsonify(error="Error processing the file"), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
