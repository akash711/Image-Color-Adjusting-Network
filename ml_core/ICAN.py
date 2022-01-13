import pickle
import random
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.callbacks import EarlyStopping, History, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam

from ml_core.batch_generator import batch_generator


class ICAN:
    history: History

    def __init__(self, model: Model, weights_path=None):
        self.model: Model = model
        weights_path and self.model.load_weights(weights_path)

    def train(self, data_path, n_samples=1000, shuffle=True, percent_train=0.8, batch_size=10,
              n_epochs=10, learn_rate=0.001, patience=10, verbose=1):
        self.model.compile(loss={
            'brightness': 'mean_squared_error',
            'contrast'  : 'mean_squared_error',
            # 'color'     : 'mean_squared_error',
            # 'sharpness' : 'mean_squared_error',
        }, optimizer=Nadam(learning_rate=learn_rate))

        index_list = [*range(n_samples)]
        shuffle and random.shuffle(index_list)

        split_index = round(percent_train * n_samples)
        train_index_list = index_list[:split_index]
        validation_index_list = index_list[split_index:]

        train_generator = batch_generator(data_path=data_path, index_list=train_index_list, batch_size=batch_size)
        validation_generator = batch_generator(data_path=data_path, index_list=validation_index_list,
                                               batch_size=batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=patience, baseline=0.01, verbose=verbose),
            TensorBoard(log_dir=join('ml_core', 'logs', 'fit', time.strftime("%Y%m%d_%H%M%S")), histogram_freq=1),
        ]
        self.history = self.model.fit(train_generator,
                                      steps_per_epoch=n_samples / batch_size, epochs=n_epochs,
                                      validation_data=validation_generator, validation_steps=n_samples / batch_size,
                                      callbacks=callbacks, verbose=verbose)

        return self.model

    def predict(self, image: Image):
        brightness, contrast = self.model.predict(np.expand_dims(image, axis=0))
        # brightness, contrast, color, sharpness = self.model.predict(np.expand_dims(image, axis=0))
        return {
            'brightness': float(brightness),
            'contrast'  : float(contrast),
            # 'color'     : float(color),
            # 'sharpness' : float(sharpness),
        }

    @staticmethod
    def adjuster(image: Image, adjustments):
        adjusted_image = image
        adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(adjustments['brightness'])
        adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(adjustments['contrast'])
        # adjusted_image = ImageEnhance.Color(adjusted_image).enhance(adjustments['color'])
        # adjusted_image = ImageEnhance.Sharpness(adjusted_image).enhance(adjustments['sharpness'])

        return adjusted_image

    def adjust(self, image: Image):
        prediction = self.predict(image)

        # Inverse prediction to get adjustments
        adjustments = {k: 1 / v for (k, v) in prediction.items()}
        adjusted_image = ICAN.adjuster(image, adjustments)

        return adjusted_image

    def match(self, image_from: Image, image_to: Image):
        prediction_from = self.predict(image_from)

        matched_image = self.adjust(image_to)
        matched_image = ICAN.adjuster(matched_image, prediction_from)

        return matched_image

    def save(self, path, model_name=None):
        model_name = model_name or self.model.name
        time_str = time.strftime("%Y%m%d_%H%M%S")  # generate timestamp

        # save JSON serialized model
        model_save_path = join(path, '%s_model_%s.json' % (model_name, time_str))
        with open(model_save_path, 'w') as model_file:
            model_file.write(self.model.to_json())
            print('Model saved to ' + model_save_path)

        # save model weights
        weights_save_path = join(path, '%s_weights_%s.h5' % (model_name, time_str))
        self.model.save_weights(weights_save_path)
        print('Model weights saved to ' + weights_save_path)

        # save History object
        history_save_path = join(path, '%s_history_%s.pickle' % (model_name, time_str))
        with open(history_save_path, 'wb') as history_file:
            pickle.dump(self.history.history, history_file)
            print('Model history saved to ' + history_save_path)

    def plot_history(self):
        if not self.history:
            print('No history found!')
            return

        plt.plot(self.history.history['loss'],
                 'r--', label='train')
        plt.plot(self.history.history['val_loss'],
                 'r-', label='validation')

        plt.plot(self.history.history['brightness_loss'],
                 'g--', label='train brightness')
        plt.plot(self.history.history['val_brightness_loss'],
                 'g-', label='validation brightness')

        plt.plot(self.history.history['contrast_loss'],
                 'b--', label='train contrast')
        plt.plot(self.history.history['val_contrast_loss'],
                 'b-', label='validation contrast')

        # plt.plot(self.history.history['color_loss'],
        #          'm--', label='train color')
        # plt.plot(self.history.history['val_color_loss'],
        #          'm-', label='validation color')

        # plt.plot(self.history.history['sharpness_loss'],
        #          'b--', label='train sharpness')
        # plt.plot(self.history.history['val_sharpness_loss'],
        #          'b-', label='validation sharpness')

        plt.title('Model loss')
        plt.ylabel('MSE loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()
