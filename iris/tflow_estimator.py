import tensorflow as tf
import numpy as np

dir(tf)
#dir(tf.summary.merge_all)


print(tf.__version__)
#print(tf.summary.merge_all)


from tensorflow.contrib.learn.python.learn.datasets import base


## Accessing the datasets.
# Data files.
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = base.load_csv_with_header(
    filename = IRIS_TRAINING,
    features_dtype = np.float32,
    target_dtype = np.int
)

test_set = base.load_csv_with_header(
    filename = IRIS_TEST,
    features_dtype = np.float32,
    target_dtype = np.int
)

# Print results.
print("\n Training_set.data")
print(training_set.data)

print("\n Test_set.data")
print(test_set.data)


## Building the model.
# Specify that all features have a  real-value data
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

classifier = tf.estimator.LinearClassifier(
    feature_columns = feature_columns,
    n_classes = 3,
    model_dir = "/tmp/iris_model"
)


## Create TensorFlow operations that generate data for the model.
# Input function.
def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        #print(features)
        #print(label)
        return features, label
    return _fn

print(input_fn(training_set)())

# raw data -> input function -> feature columns -> model


# Fit model.
classifier.train(
    input_fn = input_fn(training_set),
    steps = 1000
)
print("fit done")


# Evaluate accuracy.
accuracy_score = classifier.evaluate(
    input_fn = input_fn(test_set),
    steps = 100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

