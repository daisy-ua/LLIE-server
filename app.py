
from flask import Flask, request, jsonify
import base64
import numpy as np
from processing_module import ProcessingModule
import cv2

app = Flask(__name__)

module = ProcessingModule()

# method [retinex, non_retinex]

@app.route('/process', methods=['POST'])
def process_retinex_based():
    method = request.args.get('method')
    print(method)


    image_data = request.data
    
    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if method == "retinex":
        result = module.run_retinex_based_method(img)
    else:
        result = module.run_processing_based_method(img)

    _, img_encoded = cv2.imencode('.jpg', result)

    return base64.b64encode(img_encoded).decode('utf-8')

