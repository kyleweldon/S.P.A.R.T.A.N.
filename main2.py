#############################################
# File created by Kyle Weldon on 05/22/2024 #
#############################################

################# Needs Doing list for entire project ##############
# TODO: Try clustering the data from the total combined values and see how they relate
#############################################################

import os
# Supress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from sklearn.cluster import KMeans

class Data:

    def __init__(self, decision='3a', sums=True):
        self.Big5_train = pd.read_csv('Data/Big5Train.csv').to_numpy()
        self.Big5_val = pd.read_csv('Data/Big5Val.csv').to_numpy()

        self.DMS_train = pd.read_csv('Data/DMSTrain.csv').to_numpy()
        self.DMS_val = pd.read_csv('Data/DMSVal.csv').to_numpy()

        self.Scores_train = pd.read_csv('Data/ScoresTrain.csv').to_numpy()
        self.Scores_val = pd.read_csv('Data/ScoresVal.csv').to_numpy()

        self.Purple11_train = pd.read_csv('Data/Purple11Train.csv').to_numpy()
        self.Purple11_val = pd.read_csv('Data/Purple11Val.csv').to_numpy()

        path = f"Data/{decision}Train.csv"
        self.Y_train = pd.read_csv(path).to_numpy()
        path = f"Data/{decision}Val.csv"
        self.Y_val = pd.read_csv(path).to_numpy()

        self.X_train = np.array(None)

        if sums:
            Big5_train = np.sum(self.Big5_train, axis=1).reshape(200, 1)
            Big5_val = np.sum(self.Big5_val, axis=1).reshape(28, 1)

            DMS_train = np.sum(self.DMS_train, axis=1).reshape(200, 1)
            DMS_val = np.sum(self.DMS_val, axis=1).reshape(28, 1)

            Scores_train = np.sum(self.Scores_train, axis=1).reshape(200, 1)
            Scores_val = np.sum(self.Scores_val, axis=1).reshape(28, 1)

            Purple11_train = np.sum(self.Purple11_train, axis=1).reshape(200, 1)
            Purple11_val = np.sum(self.Purple11_val, axis=1).reshape(28, 1)

            self.X_train = np.concatenate((Big5_train, DMS_train, Scores_train, Purple11_train), axis=1)

    def build_X_data_from_clusters(self, Big5_train_labels, DMS_train_labels, Scores_train_labels, Purple11_train_labels):
        X = []
        for i in range(len(Big5_train_labels)):
            temp = []
            temp.append(Big5_train_labels[i])
            temp.append(DMS_train_labels[i])
            temp.append(Scores_train_labels[i])
            temp.append(Purple11_train_labels[i])
            X.append(temp)

        self.X_train = np.array(X)

class ArtificialIntelligence(Data):
    def centroid_clustering(self):

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(self.Big5_train)
        # centroids = kmeans.cluster_centers_
        Big5_train_labels = kmeans.labels_

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(self.DMS_train)
        # centroids = kmeans.cluster_centers_
        DMS_train_labels = kmeans.labels_

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.Scores_train)
        # centroids = kmeans.cluster_centers_
        Scores_train_labels = kmeans.labels_

        kmeans = KMeans(n_clusters=6)
        kmeans.fit(self.Purple11_train)
        # centroids = kmeans.cluster_centers_
        Purple11_train_labels = kmeans.labels_

        # temp1 = list(DMS_train_labels)
        # temp2 = list(self.Y_train)
        # correct = 0
        # for i in range(200):
        #     if temp1[1] == temp2[i][0]:
        #         correct += 1
        #     print(f"{temp1[i]}\t{temp2[i][0]}")
        # avg = correct / 200
        # print(f"Grade: {avg*100}")

        # print(Big5_train_labels)
        # print(DMS_train_labels)
        # print(Scores_train_labels)
        # print(Purple11_train_labels)

        super().build_X_data_from_clusters(Big5_train_labels, DMS_train_labels, Scores_train_labels, Purple11_train_labels)


    def build_and_compile_model(self):

        input_layer = Input(shape=(4,))
        hidden_layer1 = Dense(128, activation='relu')(input_layer)
        output = Dense(1, activation='sigmoid')(hidden_layer1)
        self.model = Model(inputs=input_layer, outputs=output)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        if self.X_train.size == 1:
            print('No training data.')
            exit(69)

        print('Input:\t\t\t\t\tExpected Output:')
        temp1 = self.X_train.tolist()
        temp2 = self.Y_train.tolist()
        for i in range(200):
            print(f"{temp1[i][0]} {temp1[i][1]} {temp1[i][2]} {temp1[i][3]}\t\t\t{temp2[i][0]}")

        self.model.fit(self.X_train, self.Y_train,
                       batch_size=200, epochs=10, verbose=1, validation_split=0.2)

if __name__ == '__main__':
    ai = ArtificialIntelligence()
    #ai.centroid_clustering()
    ai.build_and_compile_model()
    ai.train_model()
