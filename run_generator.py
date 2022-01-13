import re
import time
from os import listdir
from os.path import isfile, join

from PIL import Image

from ml_core import models, predict
from ml_core.train_generator import train

checkpoints_path = 'ml_core/checkpoints'

# Train the model
model = models.ican_mini()
model.summary()
trained_model = train(model=model,
                      data_path='data/synthesized_data',
                      n_samples=10,
                      percent_train=0.8,
                      batch_size=1,
                      n_epochs=1,
                      learn_rate=0.01)
trained_model.save_weights(join(checkpoints_path, 'ican_mini_weights_' + time.strftime("%Y%m%d_%H%M%S") + '.h5'))

# Test color adjustment
weight_iles = [f for f in listdir(checkpoints_path)f
                if isfile(join(checkpoints_path, f)) and re.match(r'ican_mini_weights_.*\.h5', f)]
model.load_weights(join(checkpoints_path, max(weight_files)))  # load latest weights

test_image = Image.open('data/synthesized_data/im83_b1.3802_c0.4381.jpg')
adjusted_image = predict.adjust(model, test_image)
adjusted_image.show()
adjusted_image.save('output/im83_adjusted.jpg')
