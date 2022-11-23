'''
conceptualize brand ne claification project here

Idea: BMI Category predictor

Prompt: Given a persons weight, height, and BMI, predict that person's BMI category
Method: Randomly generated data, classification (tensorflow)

'''

from __future__ import absolute_import, division, print_function, unicode_literals
from msilib.schema import FeatureComponents

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
import os

def input_function(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# os.system('python datagen.py')

COLUMN_NAMES = ["height_cm", "weight_kg", "bmi", "category"]
CATEGORIES = ["Underweight", "Healthy", "Overweight", "Obese", "Morbidly_Obese"]

train = pd.read_csv("data/train.csv", names=COLUMN_NAMES, header=0)
test = pd.read_csv("data/eval.csv", names=COLUMN_NAMES, header=0)

# read the error tags and figure out what's wrong

train_y = train.pop('category')
test_y = test.pop('category')

my_feature_columns = []

for key in train.keys():
  my_feature_columns.append(tf.feature_column.numeric_column(key=key))
# problem lies here
# dtype should be integer, not string
print(my_feature_columns)

# experiment wwwwwwiwth n classssesss
# watch tutorial about estimateors
# trying to use linear classifier instead of dnnclassifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=5
)

# wtf is going on lmao
# ok, if all hope is lost with this project, just copy the flower thing onto this computer

classifier.train(
    input_fn=lambda: input_function(train, train_y, training=True), 
    steps=5000
)

eval_result = classifier.evaluate(input_fn=lambda: input_function(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def input_function(features, batch_size=256):
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ["height_cm", "weight_kg", "bmi"]
predict = {}

print("Please type in values as prompted: ")

for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn = lambda: input_function(predict))

for prediction_dict in predictions:
  print(prediction_dict)
  class_id = prediction_dict['class_ids'][0]
  probability = prediction_dict['probabilities'][class_id]
  print("Main Prediction: {} ({:.1f})%".format(CATEGORIES[class_id], 100 * probability))
  print("Predictions for all/other flowers: ")
  for i in range(len(CATEGORIES)):
    print("{}: ({:.1f})%".format(CATEGORIES[i], 100 * prediction_dict['probabilities'][i]))
