import re
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml_core import models
from ml_core.ICAN import ICAN

# TensorBoard server start terminal command
# tensorboard --logdir ml_core/logs/fit

data_path = join('data', 'synthesized_data')
checkpoints_path = join('ml_core', 'checkpoints')

# Train the model
ican = ICAN(models.ican_mini())
ican.model.summary()
trained_model = ican.train(data_path=data_path,
                           n_samples=100, percent_train=0.8,
                           batch_size=10, n_epochs=100,
                           learn_rate=0.01, patience=10,
                           verbose=1)
ican.plot_history()
ican.save(path=checkpoints_path)

# Test color adjustment
test_filename = 'im83_b1.3802_c0.4381'
test_image = Image.open(join('data', 'synthesized_data', '%s.jpg' % test_filename))

plot_dpi = 100
plt.figure(figsize=[test_image.size[0] / plot_dpi, 3 * test_image.size[1] / plot_dpi], dpi=plot_dpi)
plt.subplot(311)
plt.imshow(np.array(test_image))
plt.axis('off')

# current best model with weights
ican = ICAN(model=models.ican_mini(), weights_path=join(checkpoints_path, 'ican_mini_weights_20200322.h5'))
adjusted_image = ican.adjust(test_image)

plt.subplot(312)
plt.imshow(np.array(adjusted_image))
plt.axis('off')

# latest model with weights
weight_files = [f for f in listdir(checkpoints_path)
                if isfile(join(checkpoints_path, f)) and re.match(r'ican_mini_weights_.*\.h5', f)]

ican = ICAN(model=models.ican_mini(), weights_path=join(checkpoints_path, max(weight_files)))
adjusted_image = ican.adjust(test_image)

plt.subplot(313)
plt.imshow(np.array(adjusted_image))
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

# adjusted_image.save(join('output', '%s_adjusted.jpg' % test_filename))
