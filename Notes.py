# Load datasets
training_data = load_csv_with_header()

# Define input functions
def input_fn(datasets)

# Define feature columns
feature_columns = [tf.feature_colums.numberic_column(feature_name, shape = [4])]


# Create model
classifier = tf.estimator.LinearClassifier()

# Train
classifier.train()

# Evaluate
classifier.evaluate()
