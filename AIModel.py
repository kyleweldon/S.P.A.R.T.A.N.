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
from tensorflow.keras import layers, regularizers
import numpy as np
import shap
import MeaningsAtIndecies
from Data.DataConnection import DataForTraining

class BaseModel(DataForTraining):
    def single_input_model(self):
        self.x_train = self.sequential_x_train
        self.x_validate = self.sequential_x_validate
        self.epochs = 75

        input = Input(shape=(24,), name='input')
        dense = Dense(8, activation='relu', name='dense')(input)
        output = Dense(1, activation='sigmoid', name='output')(dense)
        self.model = Model(inputs=input, outputs=output)

    def multi_input_model(self):
        self.x_train = [self.big5_x_train, self.layer2_x_train,
                    self.layer3_x_train, self.layer4_x_train]
        self.x_validate = [self.big5_x_validate, self.layer2_x_validate,
                           self.layer3_x_validate, self.layer4_x_validate]
        self.epochs = 5

        # Define input shapes for each input
        input_shape_1 = (5,)
        input_shape_2 = (5,)
        input_shape_3 = (3,)
        input_shape_4 = (11,)

        # Define input layers
        input_1 = Input(shape=input_shape_1, name='Big5')
        input_2 = Input(shape=input_shape_2, name='input_2')
        input_3 = Input(shape=input_shape_3, name='input_3')
        input_4 = Input(shape=input_shape_4, name='input_4')

        # Define dense layers for each input
        dense_1 = Dense(8, activation='relu', name='dense_1')(input_1)
        dense_2 = Dense(8, activation='relu', name='dense_2')(input_2)
        dense_3 = Dense(4, activation='relu', name='dense_3')(input_3)
        dense_4 = Dense(20, activation='relu', name='dense_4')(input_4)

        # Concatenate the outputs of the dense layers
        concatenated = concatenate([dense_1, dense_2, dense_3, dense_4])

        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(concatenated)

        # Define the model
        self.model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.x_train, self.y_train,
                      batch_size=32, epochs=self.epochs, verbose=self.verbose, validation_split=.2)

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
        print(f"Model Loss: {loss:.4f}\nModel Accuracy: {(accuracy * 100):.2f}%")

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

    def __del__(self):
        print('BaseModel object is released.')
        super().__del__()

if __name__ == '__main__':
    print('This is not the "main.py" file. Please run the correct file.')


