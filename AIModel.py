############################################
# File created by Kyle Weldon on 04/07/2024 #
#############################################

import os
# Supress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.cluster import KMeans
import numpy as np
import shap
import MeaningsAtIndecies
from Data.DataConnection import DataForTraining
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch


class BaseModel(DataForTraining):

    def single_input_model(self):
        self.x_train = tf.constant(self.sequential_x_train)
        self.x_validate = tf.constant(self.sequential_x_validate)
        self.y_train = tf.constant(self.y_train)
        self.epochs = 20

        input_layer = Input(shape=(24,))
        hidden_layer1 = Dense(64, activation='relu')(input_layer)
        hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)
        hidden_layer3 = Dense(16, activation='relu')(hidden_layer2)
        output = Dense(1, activation='sigmoid')(hidden_layer3)
        self.model = Model(inputs=input_layer, outputs=output)

    def multi_input_model(self, activiations, units):
        self.x_train =  [tf.constant(self.layer4_x_train),
                   tf.constant(self.big5_x_train),
                   tf.constant(self.layer2_x_train),
                   tf.constant(self.layer3_x_train)]
        self.x_validate = [tf.constant(self.layer4_x_validate),
                 tf.constant(self.big5_x_validate),
                 tf.constant(self.layer2_x_validate),
                 tf.constant(self.layer3_x_validate)]
        self.y_train = tf.constant(self.y_train)
        self.y_validate = tf.constant(self.y_validate)
        self.epochs = 15

        # Define input shapes for each input
        input_shape_1 = (11,)
        input_shape_2 = (5,)
        input_shape_3 = (5,)
        input_shape_4 = (3,)

        # Define input layers
        input_1 = Input(shape=input_shape_1, name='Big5')
        input_2 = Input(shape=input_shape_2, name='input_2')
        input_3 = Input(shape=input_shape_3, name='input_3')
        input_4 = Input(shape=input_shape_4, name='input_4')

        # Define dense layers for each input
        dense_1 = Dense(units[0], activation=activiations[0], name='dense_1')(input_1)
        dense_2 = Dense(units[1], activation=activiations[1], name='dense_2')(input_2)
        dense_3 = Dense(units[2], activation=activiations[2], name='dense_3')(input_3)
        dense_4 = Dense(units[3], activation=activiations[3], name='dense_4')(input_4)

        # Concatenate the outputs of the dense layers
        concatenated = concatenate([dense_1, dense_2, dense_3, dense_4])

        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(concatenated)

        # Define the model
        self.model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        hist = self.model.fit(self.x_train, self.y_train,
                      batch_size=32, epochs=self.epochs, verbose=self.verbose)
        best_score = max(hist.history['accuracy'])

        return best_score

    def explain_model(self):
        # Returns a hash table - keys are the index and value is the phycological meaning
        meanings = MeaningsAtIndecies.get_input_meanings()
        explainer = shap.Explainer(self.model, self.x_train)
        # Explain the decision for all samples
        shap_values = explainer(self.x_train,  max_evals=5000)
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        # Sort features by mean absolute SHAP values
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        # Get features sorted by mean absolute SHAP values
        sorted_features = sorted_indices.tolist()

        print("Displaying what impacted the models decisions from most to least...")
        counter = 0
        for feature in sorted_features:
            counter += 1
            print(f"{counter}: (Index {feature}) {meanings[feature]}")

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.x_validate, self.y_validate, verbose=self.verbose)
        #print(f"Model Loss: {loss:.4f}\nModel Accuracy: {(accuracy * 100):.2f}%")
        return accuracy

    def predict_with_model(self):
        prediction = self.model.predict(self.x_validate, verbose=self.verbose)
        num_of_predictions = len(prediction)
        correct_predictions = 0
        for i in range(num_of_predictions):
            print(f"Model prediction: {1 if prediction[i][0] > 0.5 else 0} -> "
                  f"Actual value: {self.y_validate[i][0]}")
            # If the prediction is less than 0.5 and actual is 0 or prediction is grater than 0.5 and the actual is 1
            if ((prediction[i][0] < .5 and self.y_validate[i][0] == 0) or
                    (prediction[i][0] > .5 and self.y_validate[i][0] == 1)):
                correct_predictions += 1
        grade = float(correct_predictions / num_of_predictions)
        print(f"The models grade is: {grade * 100.0}")

    def clustering(self):

        X = np.array(self.big5_x_train)

        # Creating KMeans instance
        kmeans = KMeans(n_clusters=4)

        # Fitting the model
        kmeans.fit(X)

        # Getting the cluster centers and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Print cluster centers
        print("Centroid locations:")
        print(centers)

        # Print cluster labels
        print("\nCluster Labels:")
        print(labels)


    def __del__(self):
        print('-------------------------------')
        super().__del__()

class Old_KMeans(DataForTraining):

    def fit(self):
        self.n_clusters = 3
        self.max_iters = 100
        X = np.array(self.big5_x_train)
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        random_indices = tf.random.shuffle(tf.range(n_samples))
        centroids = tf.gather(X, random_indices[:self.n_clusters])

        for _ in range(self.max_iters):

            # Assign each sample to the nearest centroid
            distances = tf.reduce_sum((tf.expand_dims(X, 0) - centroids) ** 2, axis=-1)
            labels = tf.argmin(distances, axis=1)

            # Update centroids
            new_centroids = tf.concat([
                tf.reduce_mean(tf.gather(X, tf.where(tf.equal(labels, i))), axis=0, keepdims=True)
                for i in range(self.n_clusters)
            ], axis=0)

            # Check convergence
            if tf.reduce_all(tf.equal(new_centroids, centroids)):
                break

            centroids = new_centroids

        self.labels_ = labels.numpy()
        self.cluster_centers_ = centroids.numpy()

class HyperperameterTuner(DataForTraining):
    def build_model(self, hp):
        '''
        This function does not get used anymore!!!
        Keeping it for now so code is not lost.
        '''
        activation_functions = [
            'elu',
            'exponential',
            'gelu',
            'hard_sigmoid',
            'linear',
            'relu',
            'selu',
            'sigmoid',
            'softmax',
            'softplus',
            'softsign',
            'swish',
            'tanh'
        ]

        # Define input shapes for each input
        input_shape_1 = (11,)
        input_shape_2 = (5,)
        input_shape_3 = (5,)
        input_shape_4 = (3,)

        # Define input layers
        input_1 = Input(shape=input_shape_1, name='Big5')
        input_2 = Input(shape=input_shape_2, name='input_2')
        input_3 = Input(shape=input_shape_3, name='input_3')
        input_4 = Input(shape=input_shape_4, name='input_4')

        # Define dense layers for each input with hyperparameters
        dense_1 = Dense(
            units=hp.Int('units_1', min_value=5, max_value=500, step=5),
            activation=hp.Choice('activation_1', values=activation_functions),
            name='dense_1'
        )(input_1)
        dense_2 = Dense(
            units=hp.Int('units_2', min_value=5, max_value=500, step=5),
            activation=hp.Choice('activation_2', values=activation_functions),
            name='dense_2'
        )(input_2)
        dense_3 = Dense(
            units=hp.Int('units_3', min_value=5, max_value=500, step=5),
            activation=hp.Choice('activation_3', values=activation_functions),
            name='dense_3'
        )(input_3)
        dense_4 = Dense(
            units=hp.Int('units_4', min_value=5, max_value=500, step=5),
            activation=hp.Choice('activation_4', values=activation_functions),
            name='dense_4'
        )(input_4)

        # Concatenate the outputs of the dense layers
        concatenated = concatenate([dense_1, dense_2, dense_3, dense_4])

        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(concatenated)

        # Define the model
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def tuner(self):
        '''
        This function does not get used anymore!!!
        Keeping it for now so code is not lost.
        '''
        # Initialize the tuner
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=100,
            executions_per_trial=5,
            directory='delme',
            project_name='multi_input_tuning'
        )

        x_train = [tf.constant(self.layer4_x_train),
                   tf.constant(self.big5_x_train),
                   tf.constant(self.layer2_x_train),
                   tf.constant(self.layer3_x_train)]
        y_train = tf.constant(self.y_train)

        x_val = [tf.constant(self.layer4_x_validate),
                 tf.constant(self.big5_x_validate),
                 tf.constant(self.layer2_x_validate),
                 tf.constant(self.layer3_x_validate)]
        y_val = tf.constant(self.y_validate)

        # Perform hyperparameter search
        tuner.search(
            x_train, y_train,
            epochs=10,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Retrieve the best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the best model
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))

        test_loss, test_acc = model.evaluate(x_val, y_val)
        print('Test accuracy:', test_acc)

        print("The best hyperparameters are:")
        print(f"Units for dense_1: {best_hps.get('units_1')}")
        print(f"Activation for dense_1: {best_hps.get('activation_1')}")
        print(f"Units for dense_2: {best_hps.get('units_2')}")
        print(f"Activation for dense_2: {best_hps.get('activation_2')}")
        print(f"Units for dense_3: {best_hps.get('units_3')}")
        print(f"Activation for dense_3: {best_hps.get('activation_3')}")
        print(f"Units for dense_4: {best_hps.get('units_4')}")
        print(f"Activation for dense_4: {best_hps.get('activation_4')}")
        print(f"Learning rate: {best_hps.get('learning_rate')}")

    def __del__(self):
        print('Perameter model has been released.')
        super().__del__()

if __name__ == '__main__':
    print('This is not the "main.py" file. Please run the correct file.')


