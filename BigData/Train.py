#############################################
# File created by Kyle Weldon on 06/16/2024 #
#############################################

################# Needs Doing list for entire project ##############
# TODO: Rename variables so they all make sense/fully organize code
# TODO: Add L2 Regularization
# TODO: Instead of inheriting the class data, have it create its own instantiation of it
####################################################################

import os
# Supress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.cluster import KMeans

def train():
    ai_obj = ArtificialIntelligence(model_type='Scenario 1 ',
                                    print_predictions=True,
                                    write_to_file=False)


class Data():
    def __init__(self, model_type, training_model=False, raw_data=False):
        if raw_data:
            self.excel_file = 'Data/RawData.xlsx'
        self.output_csv = 'Data/FilteredData.csv'

        if training_model:
            self.model_type = model_type
            df = pd.read_csv(self.output_csv)

            ######################
            # Gathering X values #
            ######################
            # input_columns = ['Q105_1','Q105_2','Q105_3','Q105_4','Q105_5','Q105_6','Q105_7','Q105_8','Q105_9','Q105_10','Q105_11','Q105_12','Q105_13','Q105_14','Q105_15','Q105_16','Q105_17','Q105_18','Q105_19','Q105_20','Q105_21','Q105_22','Q105_23','Q105_24','Q105_25','Q105_26','Q105_27','Q105_28','Q105_29','Q105_30','Q105_31','Q105_32','Q105_33','Q105_34']
            input_columns = ['MAx1', 'Max2', 'Max3']
            input_df = df[input_columns]
            all_X_values = input_df.to_numpy()

            self.X_train = all_X_values[:800]
            self.X_val = all_X_values[800:]



            ######################
            # Gathering Y values #
            ######################
            output_columns = ['Scenario 1 ',
                              'Unnamed: 40',
                              'Scenario 2 ',
                              'Unnamed: 42',
                              'Scenario 3 ',
                              'Unnamed: 44',
                              'Scenario 4',
                              'Unnamed: 46',
                              'Scenario 5 ',
                              'Unnamed: 48']


            # For multi-ouput model
            if self.model_type.lower() == 'all':
                self.single = False
                multi_output_df = df[output_columns]

                all_Y_values = multi_output_df.to_numpy()

                labeled_Y_values = []
                for each in all_Y_values:
                    temp = []
                    for every in each:
                        if every <= 3:
                            temp.append(0)
                        elif every <= 6:
                            temp.append(1)
                        else:
                            temp.append(2)
                    labeled_Y_values.append(temp)

                labeled_Y_values = np.array(labeled_Y_values)

                y_train_categorized = to_categorical(labeled_Y_values, num_classes=3)

                self.multi_Y_train = y_train_categorized[:800]
                self.multi_Y_val = y_train_categorized[800:]

            # For single ouput model
            elif self.model_type in output_columns:
                self.single = True
                single_output_df = df[self.model_type]

                all_Y_values = single_output_df.to_numpy()

                labeled_Y_values = []
                for decision in all_Y_values:
                    if decision <= 3:
                        labeled_Y_values.append(0)
                    elif decision <= 6:
                        labeled_Y_values.append(1)
                    else:
                        labeled_Y_values.append(2)

                labeled_Y_values = np.array(labeled_Y_values)

                y_train_categorized = to_categorical(labeled_Y_values, num_classes=3)

                self.single_Y_train = y_train_categorized[:800]
                self.single_Y_val = y_train_categorized[800:]

            else:
                print('\n\nInvalid model type has been entered. Ending program now.\n')
                exit(222)

    def filter_data(self):
        try:
            df = pd.read_excel(self.excel_file)
        except FileNotFoundError:
            print(f"Error: The file '{self.excel_file}' was not found.")
            return
        except Exception as e:
            print(f"Error occurred while reading '{self.excel_file}': {str(e)}")
            return

        # Step 2: Filter rows based on completeness (non-empty cells)
        complete_rows = []
        for index, row in df.iterrows():
            if self.is_row_complete(row):
                complete_rows.append(row)

        cleaned_df = pd.DataFrame(complete_rows, columns=df.columns)

        # Step 3: Save the cleaned data to a CSV file
        try:
            cleaned_df.to_csv(output_csv, index=False)
            print(f"Cleaned data saved to '{self.output_csv}' successfully.")
        except Exception as e:
            print(f"Error occurred while saving to '{self.output_csv}': {str(e)}")
            return

    def is_row_complete(self, row):
        for cell in row:
            # Check if cell is NaN or empty (after stripping whitespace)
            if pd.isna(cell) or str(cell).strip() == '':
                return False
        return True

    def write_multi_output_to_csv(self, predictions):
        headers = ['Person',
                   'Scenario 1 pt1', 'Scenario 1 pt2',
                   'Scenario 2 pt1', 'Scenario 2 pt2',
                   'Scenario 3 pt1', 'Scenario 3 pt2',
                   'Scenario 4 pt1', 'Scenario 4 pt2',
                   'Scenario 5 pt1', 'Scenario 5 pt2']

        num = len(predictions)
        predictions_df = pd.DataFrame({'Person': [i+1 for i in range(num)],
                                       'Scenario 1 pt1': [f"{predictions[i][0]} -> {self.multi_Y_val[i][0]}" for i in range(num)],
                                       'Scenario 1 pt2': [f"{predictions[i][1]} -> {self.multi_Y_val[i][1]}" for i in range(num)],
                                       'Scenario 2 pt1': [f"{predictions[i][2]} -> {self.multi_Y_val[i][2]}" for i in range(num)],
                                       'Scenario 2 pt2': [f"{predictions[i][3]} -> {self.multi_Y_val[i][3]}" for i in range(num)],
                                       'Scenario 3 pt1': [f"{predictions[i][4]} -> {self.multi_Y_val[i][4]}" for i in range(num)],
                                       'Scenario 3 pt2': [f"{predictions[i][5]} -> {self.multi_Y_val[i][5]}" for i in range(num)],
                                       'Scenario 4 pt1': [f"{predictions[i][6]} -> {self.multi_Y_val[i][6]}" for i in range(num)],
                                       'Scenario 4 pt2': [f"{predictions[i][7]} -> {self.multi_Y_val[i][7]}" for i in range(num)],
                                       'Scenario 5 pt1': [f"{predictions[i][8]} -> {self.multi_Y_val[i][8]}" for i in range(num)],
                                       'Scenario 5 pt2': [f"{predictions[i][9]} -> {self.multi_Y_val[i][9]}" for i in range(num)]})

        predictions_df.to_csv('Data/predictions.csv', index=False)
        print(f"Predictions saved to 'Data/predictions.csv' successfully.")

    def __del__(self):
        print()

class ArtificialIntelligence():
    '''
    Explain this class here.
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data_obj = Data(model_type=self.kwargs['model_type'], training_model=True)

        # Calling function to train the model
        if self.data_obj.single:
            self.single_output_model()
        elif not self.data_obj.single:
            self.multi_output_model()
        else:
            print('\n\nInvalid model type has been entered. Ending program now.\n')
            exit(333)

    def multi_output_model(self):
        inputs = Input(shape=(self.data_obj.X_train.shape[1],))

        hidden1 = Dense(256, activation='relu')(inputs)
        hidden2 = Dense(128, activation='relu')(hidden1)
        hidden3 = Dense(64, activation='relu')(hidden2)
        hidden4 = Dense(32, activation='relu')(hidden3)
        hidden5 = Dense(16, activation='relu')(hidden4)

        outputs = []
        for i in range(10):
            output = Dense(3, activation='softmax', name=f'output_{i + 1}')(hidden5)
            outputs.append(output)

        reshaped_outputs = [Reshape((1, 3))(output) for output in outputs]
        concatenated_outputs = Concatenate(axis=1)(reshaped_outputs)

        self.model = Model(inputs=inputs, outputs=concatenated_outputs)

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(self.data_obj.X_train, self.data_obj.multi_Y_train, epochs=10, batch_size=32)


        predictions = self.model.predict(self.data_obj.X_val, verbose=0)
        if self.kwargs['print_predictions']:
            for i in range(len(predictions)):
                print(f"Person {i + 1}:")
                for j in range(len(predictions[i])):
                    print(f"\tScenario {j + 1}:")
                    print(f"\t\t{predictions[i][j]} -> {self.data_obj.multi_Y_val[i][j]}")

        if self.kwargs['write_to_file']:
            self.data_obj.write_multi_output_to_csv(predictions)

    def single_output_model(self):
        inputs = Input(shape=(self.data_obj.X_train.shape[1],))

        hidden1 = Dense(256, activation='relu')(inputs)
        hidden2 = Dense(128, activation='relu')(hidden1)
        hidden3 = Dense(64, activation='relu')(hidden2)
        hidden4 = Dense(32, activation='relu')(hidden3)
        hidden5 = Dense(16, activation='relu')(hidden4)


        output = Dense(3, activation='softmax')(hidden5)

        self.model = Model(inputs=inputs, outputs=output)

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(self.data_obj.X_train, self.data_obj.single_Y_train, epochs=10, batch_size=32)

        predictions = self.model.predict(self.data_obj.X_val, verbose=0)
        for i in range(len(predictions)):
            print(f"{predictions[i]} -> {self.data_obj.single_Y_val[i]}")

    def __del__(self):
        print()

if __name__ == '__main__':
    if False:
        train()

    if True:
        pass
