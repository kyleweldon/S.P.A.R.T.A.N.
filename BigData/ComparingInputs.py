#############################################
# File created by Kyle Weldon on 06/16/2024 #
#############################################

'''
Most of this code was copied from the "CurrentWorkspace.ipynb" file.
That file is also created and maintained by Kyle Weldon.
'''

import os
# Supress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pprint

import warnings
# Supress a specific TensorFlow warning (Warning not to use a feature I do not use)
warnings.filterwarnings('ignore', category=UserWarning, message="Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported. See PR #1483 for discussion.")

import datetime

import numpy as np
import pandas as pd

from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from tensorboard import program
from tensorflow.keras.utils import plot_model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import shap

def main():
    options = ['maxs', 'quests']
    max_labels = ['max1', 'max2', 'max3']
    questionset_labels = ['Q105_1', 'Q105_2', 'Q105_3', 'Q105_4', 'Q105_5', 'Q105_6', 'Q105_7', 'Q105_8', 'Q105_9', 'Q105_10',
              'Q105_11', 'Q105_12', 'Q105_13', 'Q105_14', 'Q105_15', 'Q105_16', 'Q105_17', 'Q105_18', 'Q105_19',
              'Q105_20', 'Q105_21', 'Q105_22', 'Q105_23', 'Q105_24', 'Q105_25', 'Q105_26', 'Q105_27', 'Q105_28',
              'Q105_29', 'Q105_30', 'Q105_31', 'Q105_32', 'Q105_33', 'Q105_34']
    calculations = {'max1': questionset_labels[0:11],
                    'max2': questionset_labels[11:23],
                    'max3': questionset_labels[23:34]}

    # Running the model and info to lists
    results = []
    shap_stuff = []
    for type in options:
        ai_obj = Artificial_Intellegence(type)
        results.append(ai_obj.train())
        shap_stuff.append(ai_obj.shap_EXplination())
        ai_obj = None # Set to be deleted by python if space needed

    # Comparing the models
    maxs_results = results[0]
    quests_results = results[1]
    table = [['Situation', 'Maxs', 'Question set']]
    for key in maxs_results.keys(): # They have the same keys becuase they are predicting the same output
        temp = []
        temp.append(key)
        temp.append(maxs_results[key][-1] if 'loss' in key else f'{round(maxs_results[key][-1]*100, 2)}%')
        temp.append(quests_results[key][-1] if 'loss' in key else f'{round(quests_results[key][-1]*100, 2)}%')
        table.append(temp)

    print(tabulate(table, headers='firstrow', tablefmt='grid'))

    print('\t'*50)

    # Dysplaying shap information
    print('Keys:')
    for key, value in calculations.items():
        print(f'\t{key}: {value}')

    maximization_shaps = shap_stuff[0]
    questionset_shaps = shap_stuff[1]
    # if maximization_shaps.keys() == questionset_shaps.keys():
    for key in maximization_shaps.keys() if maximization_shaps.keys() == questionset_shaps.keys() else []:
        print(f'{key}:')
        print('\tMaximizations:')
        for place, i in enumerate(maximization_shaps[key]):
            print(f'\t\t{place+1}: {max_labels[i]}')
        print('\tQuestionset:')
        for place, i in enumerate(questionset_shaps[key]):
            print(f'\t\t{place+1}: {questionset_labels[i]}')

        # for f, i in enumerate(maximization_shaps[key]):
        #     print(f'\t{f+1}: {max_labels[i]}')
        # for f, i in enumerate(questionset_shaps[key]):
        #     print(f'\t{f + 1}: {questionset_labels[i]}')

class Data:

    def __init__(self, input_type):
        self.input_type = input_type
        self.df = pd.read_csv('Data/FilteredData.csv')
        # Column tites for all the output data
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

        def classiy_and_catigorize(column):
            return to_categorical([0 if x <= 3 else 1 if x <= 6 else 2 for x in column])

        columns = self.df[output_columns].to_numpy().T  # The 'T' is to transpose the array

        S1P1, S1P2, S2P1, S2P2, S3P1, S3P2, S4P1, S4P2, S5P1, S5P2 = [classiy_and_catigorize(col) for col in columns]
        self.all_situations = [S1P1, S1P2, S2P1, S2P2, S3P1, S3P2, S4P1, S4P2, S5P1, S5P2]

    def get_data(self):
        if self.input_type == 'maxs':
            return self._maxs(), self.all_situations
        elif self.input_type == 'quests':
            return self._questionset(), self.all_situations
        else:
            print('That was not a valid input type.')
            exit(69)

    def _maxs(self):
        input_columns1 = ['MAx1', 'Max2', 'Max3']

        input_df1 = self.df[input_columns1]
        input1_X_values = input_df1.to_numpy()

        return input1_X_values

    def _questionset(self):
        input_columns2 = ['Q105_1','Q105_2','Q105_3','Q105_4','Q105_5','Q105_6','Q105_7','Q105_8','Q105_9','Q105_10','Q105_11','Q105_12','Q105_13','Q105_14','Q105_15','Q105_16','Q105_17','Q105_18','Q105_19','Q105_20','Q105_21','Q105_22','Q105_23','Q105_24','Q105_25','Q105_26','Q105_27','Q105_28','Q105_29','Q105_30','Q105_31','Q105_32','Q105_33','Q105_34']

        input_df2 = self.df[input_columns2]
        input2_X_values = input_df2.to_numpy()

        return input2_X_values

class Artificial_Intellegence:

    def __init__(self, input_type):
        data_obj = Data(input_type)
        self.x_train, self.y_train = data_obj.get_data()

        input = Input(shape=(self.x_train.shape[1],))

        hidden1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(input)
        hidden2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(hidden1)
        hidden3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(hidden2)
        hidden4 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(hidden3)
        hidden5 = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(hidden4)

        def create_output_layer(name, input_layer):
            return Dense(3, activation='softmax', name=name)(input_layer)

        def itterate_situations_and_parts(num_itts=10):
            scenerio = 1
            sittos = []
            for i in range(num_itts):
                part = 1 if i % 2 == 0 else 2
                sittos.append(f"S{scenerio}P{part}")
                if part == 2: scenerio += 1
            return sittos

        outputs = [create_output_layer(name, hidden5) for name in itterate_situations_and_parts()]

        self.model = Model(inputs=input, outputs=outputs, name='CustomizedDeepNeuralNetwork')

        metrics = ['accuracy'] * 10

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=metrics)

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

    def train(self):
        return self.model.fit(self.x_train, self.y_train, epochs=10,
                               batch_size=16,
                               shuffle=True,
                               verbose=0,
                               validation_split=0.2,
                              callbacks=[self.tensorboard_callback]).history

    def shap_EXplination(self):
        info = {}
        separate_models = []

        for output in self.model.outputs:
            separate_model = Model(inputs=self.model.inputs, outputs=output)
            separate_models.append(separate_model)

        background = self.x_train[:50]

        scenario = 1
        for i, separate_model in enumerate(separate_models):
            part = 1 if i % 2 == 0 else 2
            explainer = shap.DeepExplainer(separate_model, background)
            shap_values = explainer.shap_values(self.x_train)
            # shap.summary_plot(shap_values, self.x_train,
            #                   plot_type='bar', show=True)
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
            mean_again = np.mean(mean_abs_shap_values, axis=1)
            sorted_indices = np.argsort(mean_again)[::-1]
            key = f"S{scenario}P{part}"
            info[key] = sorted_indices.tolist()
            if part == 2: scenario += 1
        return info

            # plt.figure(i)
            # shap.summary_plot(shap_values, self.x_train,
            #                   plot_type='bar')
            # plt.title(f'SHAP Values for Output {i}.')
            # plt.show()

if __name__ == '__main__':
    if True:
        main()
    if False:
        options = ['maxs', 'quests']
        ai = Artificial_Intellegence(options[0])
        ai.train()
        ai.shap_EXplination()

# tensorboard --logdir C:\KyleWeldon\Projects\ThinkTank\S.P.A.R.T.A.N\BigData\logs
