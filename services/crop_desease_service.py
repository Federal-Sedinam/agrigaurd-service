import numpy as np

from disease_info import disclaimer, pepper_diseases, potato_diseases, tomato_diseases, corn_diseases

from .__common import decode_base64_image, preprocess_image_for_mobilenet_v2, preprocess_image_for_mobilenet_v3, load_model, predict


def get_disease_info(predictions, crop_type):
    max_index = np.argmax(predictions)
    confiedence = float(np.max(predictions))

    if crop_type.lower() == "corn":
        disease_info = corn_diseases[str(max_index)]
        # add confidence and disclaimer to disease_info
        disease_info["confidence"] = confiedence
        disease_info["disclaimer"] = disclaimer
        return disease_info
    elif crop_type.lower() == "pepper":
        disease_info = pepper_diseases[str(max_index)]
        # add confidence and disclaimer to disease_info
        disease_info["confidence"] = confiedence
        disease_info["disclaimer"] = disclaimer
        return disease_info
    elif crop_type.lower() == "potato":
        disease_info = potato_diseases[str(max_index)]
        # add confidence and disclaimer to disease_info
        disease_info["confidence"] = confiedence
        disease_info["disclaimer"] = disclaimer
        return disease_info
    elif crop_type.lower() == "tomato":
        disease_info = tomato_diseases[str(max_index)]
        # add confidence and disclaimer to disease_info
        disease_info["confidence"] = confiedence
        disease_info["disclaimer"] = disclaimer
        return disease_info
    else:
        raise ValueError("Invalid crop type")
        


def predict_crop_disease(base64_str: str, crop_type: str):
    image = decode_base64_image(base64_str)
    if crop_type.lower() == "corn":
        preprocessed_image = preprocess_image_for_mobilenet_v3(image)
        model_path = "./artefacts/models/corn_mobilenetsmall.h5"
        model = load_model(model_path)
        predictions = predict(model, preprocessed_image)
        disease_info = get_disease_info(predictions, crop_type)
        return disease_info
        
    else:
        if crop_type.lower() in ["pepper", "potato", "tomato"]:
            preprocessed_image = preprocess_image_for_mobilenet_v2(image)
            model_path = f"./artefacts/models/{crop_type.lower()}_mobilenetv2.h5"
            model = load_model(model_path)
            predictions = predict(model, preprocessed_image)
            disease_info = get_disease_info(predictions, crop_type)
            return disease_info
        else:
            raise ValueError("Invalid crop type")
