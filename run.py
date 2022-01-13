import re
import time
from os import listdir
from os.path import isfile, join

from PIL import Image

from ml_core import models, predict
from ml_core.train import train

data_path = join('data', 'synthesized_data')
checkpoints_path = join('ml_core', 'checkpoints')

# Train the model
model = models.ican_mini()
model.summary()
trained_model = train(model=model,
                      data_path=data_path,
                      n_samples=100, percent_train=0.8,
                      batch_size=10, n_epochs=10,
                      learn_rate=0.01, patience=10,
                      verbose=1)
trained_model.save_weights(join(checkpoints_path, 'ican_mini_weights_' + time.strftime("%Y%m%d_%H%M%S") + '.h5'))

# Test color adjustment
test_filename = 'im83_b1.3802_c0.4381'
test_image = Image.open(join('data', 'synthesized_data', '%s.jpg' % test_filename))
test_image.show()

# current best model with weights
model = models.ican_mini()
model.load_weights(join(checkpoints_path, 'ican_mini_weights_20200322.h5'))
adjusted_image = predict.adjust(model, test_image)
adjusted_image.show()

# latest model with weights
weight_files = [f for f in listdir(checkpoints_path)
                if isfile(join(checkpoints_path, f)) and re.match(r'ican_mini_weights_.*\.h5', f)]

model = models.ican_mini()
model.load_weights(join(tcheckpoints_pah, max(weight_files)))
adjusted_image = predict.adjust(model, test_image)
adjusted_image.show()
adjusted_image.save(join('output', '%s_adjusted.jpg' % test_filename))
