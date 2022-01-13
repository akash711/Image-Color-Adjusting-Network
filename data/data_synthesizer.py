import os
import random
import re
import time
from os import listdir
from os.path import isfile, join

from PIL import Image

from ml_core.ICAN import ICAN


def synthesize_data(input_images, output_images, verbose=False):
    os.makedirs(output_images['dir'], exist_ok=True)

    input_files = [f for f in listdir(input_images['dir'])
                   if isfile(join(input_images['dir'], f)) and re.match(input_images['format_re'], f)]

    verbose and print('--- BEGIN SYNTHESIZE DATA ---')
    start_time = time.time()
    for i, file in enumerate(input_files[:input_images['n_images']]):
        im = Image.open(join(input_images['dir'], file))

        adjustments = {
            'brightness': random.uniform(0.1, 2.0),
            'contrast'  : random.uniform(0.1, 2.0),
            # 'color'     : random.uniform(0.1, 2.0),
            # 'sharpness' : random.uniform(0.1, 2.0),
        }

        im = ICAN.adjuster(im, adjustments)

        im_num = int(re.findall(r'(?<=^im)\d*', file)[0])
        output_path = join(output_images['dir'], output_images['format'].format(im_num, **adjustments))
        im.save(output_path)
        verbose and print('[%d/%d] ' % (i, input_images['n_images']) + output_path)

    verbose and print('--- SYNTHESIZE DATA DONE IN %.4fs ---' % (time.time() - start_time))


if __name__ == '__main__':
    input_images = {
        'dir'      : 'mirflickr25k',
        'format_re': r'im\d*.jpg',
        'n_images' : 100,
    }

    output_images = {
        'dir'   : 'synthesized_data',
        'format': 'im{0:d}'
                  '_b{brightness:.4f}'
                  '_c{contrast:.4f}'
                  # '_k{color:.4f}'
                  # '_s{sharpness:.4f}'
                  '.jpg',
    }

    synthesize_data(input_images, output_images, verbose=True)
