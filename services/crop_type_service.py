from typing import Tuple
import json

import numpy as np

from .__common import (
    decode_base64_image, 
    preprocess_image_for_mobilenet_v2, 
    preprocess_image_for_leaf_non_leaf, 
    load_model, 
    predict
)


def predict_crop(base64_str) -> Tuple[str, float]:
    image = decode_base64_image(base64_str)
    preprocessed_image = preprocess_image_for_leaf_non_leaf(image)
    model = load_model("./artefacts/models/leaf_non_leaf.h5")
    predictions = predict(model, preprocessed_image)
    crop_index = np.argmax(predictions)
    confidence = np.max(predictions)
    if crop_index == 1:
        return "Non-Leaf", float(confidence)

    preprocessed_image = preprocess_image_for_mobilenet_v2(image)
    model = load_model("./artefacts/models/crops_mobilenetv2.h5")
    predictions = predict(model, preprocessed_image)
    with open("./crop_info.json", "r") as f:
        crop_types = json.load(f)
    crop_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return crop_types[str(crop_index)], float(confidence)