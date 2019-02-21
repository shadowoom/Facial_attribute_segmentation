import os
import codecs
import logging
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG, \
    format="%(asctime)s:%(levelname)s:%(message)s"
    )

def read_landmark_points(points_input_path):
    files = os.listdir(points_input_path)
    points = defaultdict(list)
    for f in files:
        if f.endswith('txt'):
            single_point_file = os.path.join(points_input_path, f)
            with codecs.open(single_point_file, mode='r', encoding='utf-8') as rf:
                try:
                    lines = rf.readlines()
                    file_name = lines[0].strip()
                    logging.debug('points file name: {}'.format(file_name))
                    for line in lines[1:-1]:
                        # remove front and end whitespace
                        line = re.sub(r'\s+', '', line.strip())
                        points[file_name].append(line.split(','))
                    logging.debug('points list length: {}'.format(len(points[file_name])))
                except Exception as e:
                    logging.error('error reading {}'.format(single_point_file))
                finally:
                    rf.close()
    
        
    # convert list to numpy
    points_converted = defaultdict(partial(np.ndarray, 0))
    for key, value in points.items():
        points_converted[key] = np.asarray(value, dtype=float)
        logging.debug('converted file: {} and converted type: {} and shape: {}'.format(key,type(points_converted[key]), points_converted[key].shape))
    logging.debug('converted points files number: {}'.format(len(points_converted.keys())))
    return points_converted


def visualize_points_on_image(image_path, points):
    im = plt.imread(image_path)
    logging.debug(image_path)
    implot = plt.imshow(im)
    logging.debug(points)
    scatter_plot = plt.scatter(points.T[0], points.T[1], color='red', s=1)
    scatter_plot.axes.get_xaxis().set_visible(False)
    scatter_plot.axes.get_yaxis().set_visible(False)
    plt.show()


converted_points = read_landmark_points('/home/e0397123/projects/face-parsing/dataset/SmithCVPR2013_dataset_resized/points')
image_path = '/home/e0397123/projects/face-parsing/dataset/SmithCVPR2013_dataset_resized/images/100032540_1.jpg'
#key = list(converted_points.keys())[0]
#image_path = os.path.join(image_path, key+'.jpg')
visualize_points_on_image(image_path, converted_points['100032540_1'])

