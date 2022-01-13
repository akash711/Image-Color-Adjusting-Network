import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.models import Model


def predict(model: Model, image: Image):
    brightness, contrast = model.predict(np.expand_dims(image, axis=0))
    # brightness, contrast, color, sharpness = self.model.predict(np.expand_dims(image, axis=0))
    return {
        'brightness': float(brightness),
        'contrast'  : float(contrast),
        # 'color'  : float(color),
        # 'sharpness'  : float(sharpness),
    }


def adjust(model: Model, image: Image):
    prediction = predict(model, image)

    adjusted_image = image
    adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(1 / prediction['brightness'])
    adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(1 / prediction['contrast'])
    # adjusted_image = ImageEnhance.Color(adjusted_image).enhance(1 / prediction['color'])
    # adjusted_image = ImageEnhance.Sharpness(adjusted_image).enhance(1 / prediction['sharpness'])

    return adjusted_image


def match(model: Model, image_from: Image, image_to: Image):
    prediction_from = model.predict(image_from)

    matched_image = model.adjust(image_to)
    matched_image = ImageEnhance.Brightness(matched_image).enhance(prediction_from['brightness'])
    matched_image = ImageEnhance.Contrast(matched_image).enhance(prediction_from['contrast'])
    # matched_image = ImageEnhance.Color(matched_image).enhance(prediction_from['color'])
    # matched_image = ImageEnhance.Sharpness(matched_image).enhance(prediction_from['sharpness'])

    return matched_image
