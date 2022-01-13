import random

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ml_core.batch_generator import batch_generator


def train(model: Model, data_path, n_samples=1000, shuffle=True, percent_train=0.8, batch_size=10,
          n_epochs=10, learn_rate=0.001, patience=10, verbose=1):
    model.compile(loss={
        'brightness': 'mean_squared_error',
        'contrast'  : 'mean_squared_error',
        # 'color'     : 'mean_squared_error',
        # 'sharpness' : 'mean_squared_error',
    }, optimizer=Adam(lr=learn_rate))

    index_list = [*range(n_samples)]
    shuffle and random.shuffle(index_list)

    split_index = round(percent_train * n_samples)
    train_index_list = index_list[:split_index]
    validation_index_list = index_list[split_index:]

    train_generator = batch_generator(data_path=data_path, index_list=train_index_list, batch_size=batch_size)
    validation_generator = batch_generator(data_path=data_path, index_list=validation_index_list, batch_size=batch_size)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, baseline=0.01, verbose=verbose)

    model.fit(train_generator,
              steps_per_epoch=n_samples / batch_size, epochs=n_epochs,
              validation_data=validation_generator, validation_steps=n_samples / batch_size,
              callbacks=[early_stop], verbose=verbose)

    return model
