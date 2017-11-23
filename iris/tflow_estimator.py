import tensorflow as tf
import numpy as np

dir(tf)
#dir(tf.summary.merge_all)


print(tf.__version__)
#print(tf.summary.merge_all)


from tensorflow.contrib.learn.python.learn.datasets import base

# Data files.
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets
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



