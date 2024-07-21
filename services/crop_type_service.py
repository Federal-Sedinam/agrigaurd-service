from typing import Tuple
import json

import numpy as np

from .__common import decode_base64_image, preprocess_image_for_mobilenet_v2, load_model, predict


def predict_crop(base64_str):
    image = decode_base64_image(base64_str)
    preprocessed_image = preprocess_image_for_mobilenet_v2(image)
    model = load_model("./artefacts/models/crops_mobilenetv2.h5")
    predictions = predict(model, preprocessed_image)
    return predictions

def get_crop_type(prediction) -> Tuple[str, float]:
    with open("./crop_info.json", "r") as f:
        crop_types = json.load(f)
    crop_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return crop_types[str(crop_index)], float(confidence)
