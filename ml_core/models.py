import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def ican_mini():
    img_input = layers.Input(shape=(None, None, 3))  # Arbitrary size RGB image

    # Block 1
    x = layers.Conv2D(64, (5, 5),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (5, 5),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Regression block
    # Use layers.GlobalMaxPooling2D instead of layers.Flatten because of the input of size None
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dense(1024, activation='relu', name='fc2')(x) 

    brightness = layers.Dense(1, activation='linear', name='brightness')(x)
    contrast = layers.Dense(1, activation='linear', name='contrast')(x)
    # color = layers.Dense(1, activation='linear', name='color')(x)
    # sharpness = layers.Dense(1, activation='linear', name='sharpness')(x)

    # Create model.
    model = Model(inputs=img_input, outputs=[brightness, contrast], name='ican_mini')
    # model = Model(inputs=img_input, outputs=[brightness, contrast, color, sharpness], name='ican_mini')
    return model


def ican_vgg19():
    img_input = layers.Input(shape=(None, None, 3))  # Arbitrary size RGB image

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Regression block
    # Use layers.GlobalMaxPooling2D instead of layers.Flatten because of the input of size None
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)

    brightness = layers.Dense(1, activation='linear', name='brightness')(x)
    contrast = layers.Dense(1, activation='linear', name='contrast')(x)
    # color = layers.Dense(1, activation='linear', name='color')(x)
    # sharpness = layers.Dense(1, activation='linear', name='sharpness')(x)

    # Create model.
    model = Model(inputs=img_input, outputs=[brightness, contrast], name='ican_vgg19')
    # model = Model(inputs=img_input, outputs=[brightness, contrast, color, sharpness], name='ican_mini')
    return model
