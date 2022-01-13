import re
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image


def load_image(path, padded_size=(500, 500)):
    img = np.asarray(Image.open(path))

    # zero-pad image
    padded_img = np.zeros((padded_size[0], padded_size[1], 3))
    padded_img[:img.shape[0], :img.shape[1], :3] = img

    return padded_img


def load_data(path, index_list):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    filenames = [filenames[i] for i in index_list]

    data = {
        'id'   : np.array([int(re.findall(r'(?<=^im)\d+', f)[0]) for f in filenames]),
        'data' : np.array([load_image(join(path, f)) for f in filenames]),
        'label': {
            'brightness': np.array([float(re.findall(r'(?<=_b)\d\.*\d*', f)[0]) for f in filenames]),
            'contrast'  : np.array([float(re.findall(r'(?<=_c)\d\.*\d*', f)[0]) for f in filenames]),
            # 'color'     : np.array([float(re.findall(r'(?<=_k)\d\.*\d*', f)[0]) for f in filenames]),
            # 'sharpness' : np.array([float(re.findall(r'(?<=_s)\d\.*\d*', f)[0]) for f in filenames]),
        },
    }
    return data
