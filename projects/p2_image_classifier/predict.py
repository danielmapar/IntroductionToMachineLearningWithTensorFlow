import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action = "store")
parser.add_argument('saved_model', action = "store")
parser.add_argument('--top_k', action = "store", dest = "top_k", type = int, default=5)
parser.add_argument('--category_names', action = "store", dest = "category_names")

results = parser.parse_args()
top_k = results.top_k
image_path = results.image_path
saved_model = results.saved_model
category_filename = results.category_names

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict_class(image_path, model, top_k=5):
    processed_test_image = process_image(np.asarray(Image.open(image_path)))
    img_pred = np.expand_dims(processed_test_image, axis=0)
    preds = model.predict(img_pred)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]
    return probs, classes

model = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer})

image = np.asarray(Image.open(image_path)).squeeze()
probs, classes = predict_class(image_path, model, top_k)

if category_filename != None:
    with open(category_filename, 'r') as f:
        class_names = json.load(f)
    keys = [str(x+1) for x in list(classes)]
    classes = [class_names.get(key) for key in keys]

print('Top {} classes:'.format(top_k))
for i in np.arange(top_k):
    print('Class: {}'.format(classes[i]))
    print('Probability: {:.2%}'.format(probs[i]))
    print('----------------------------------------------\n')