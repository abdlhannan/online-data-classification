from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
import pandas as pd
stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2, n_features=10, n_centroids=50)
stream.prepare_for_use()
X, Y = stream.next_sample(10000)
df = pd.DataFrame(X)
df['Y'] = Y
df.to_csv('RBF Dataset.csv', index=0, header=None)

stream = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10,
n_centroids = 50, change_speed=10, num_drift_centroids=50)
stream.prepare_for_use()
X, Y = stream.next_sample(10000)
df = pd.DataFrame(X)
df['Y'] = Y
df.to_csv('RBF Dataset 10.csv', index=0, header=None)


stream = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10,
n_centroids = 50, change_speed=70, num_drift_centroids=50)
stream.prepare_for_use()
X, Y = stream.next_sample(10000)
df = pd.DataFrame(X)
df['Y'] = Y
df.to_csv('RBF Dataset 70.csv', index=0, header=None)