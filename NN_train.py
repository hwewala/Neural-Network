#Description:
    #Script requires path to both training and test csv files since we are training based on one-hot encoded categorical data
    #So that we can standardize the number of input columns in both train and test data
#Authors: Hakeem Wewala, Harinderpal Khakh
#Last Updated: April 7, 2019

import tensorflow as tf
import numpy as np
import datetime
# pandas for reading CSV files
import pandas as pd
from pandas.api.types import CategoricalDtype
import os
import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
import sys
import csv

# np.set_printoptions(threshold=sys.maxsize)
# setting for dynamic memory allocation in GPU to avoid CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Class for storing lists of prediction accuracy for each individual output category
# Each output category should get its own class instance
class scoreList:
    def __init__ (self, s):
        self.desc = s
        self.l = []

# simplifies column values to boolean values
def simplifytoBoolean(row, strings, col):
    if any(x in row[col] for x in strings):
        boolVal = "Yes"
    elif row[col] == "Unknown":
        boolVal = "Unknown"
    else:
        boolVal = "No"
    return boolVal

# Converts age from years, months, or weeks into days
def convertAge(row):
    cell = row["AgeuponOutcome"]
    if not "Unknown" in cell:
        age = int(cell.split(" ", 1)[0])

    returnval = cell

    if "0" in cell or "Unknown" in cell:
        returnval = 0 
    elif "years" in cell or "year" in cell:
        ageInDays = age*365
        returnval = ageInDays
    elif "months" in cell or "month" in cell:
        ageInDays = age*30
        returnval = ageInDays
    elif "weeks" in cell or "week" in cell:
        ageInDays = age*7
        returnval = ageInDays
    elif "day" in cell or "days" in cell: #to make results uniform
        returnval = age

    return returnval
# Converts date time into the specific month, hour or day of the week
def convertDateTime(row, conversion):
    cell = row["DateTime"]
    date = datetime.datetime.strptime(cell, "%m/%d/%Y %H:%M")

    if conversion == "Month":
        convertedUnit = date.month
    elif conversion == "Day":
        convertedUnit = date.strftime('%d')
    elif conversion == "Hour":
        convertedUnit = date.hour
    else:
        print("Unexpected conversion in convertDateTime function")
        exit(1)
    return convertedUnit

def categoricalToNumeric(df):
    #df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "IsMix", "N/S", "DayofWeek"])
    df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "IsMix", "N/S"])


    # Re-arrange dataframe to put OutcomeType as the last column
    if 'OutcomeType' in df.columns:
        cols = list(df)
        cols.insert(len(cols), cols.pop(cols.index('OutcomeType')))
        df = df.ix[:, cols]

    print(df)
    return df

def dataManipulation(df):
    df = df.drop("Name", axis=1)
    if 'OutcomeSubtype' in df.columns:
        df = df.drop("OutcomeSubtype", axis=1)
    if 'AnimalID' in df.columns:
        df.rename(columns={'AnimalID' : 'ID'}, inplace=True)

    df["Color"] = df["Color"].replace("\/.*", "", regex=True) # Cutting everything after / in Color column
    df.rename(columns={'Color' : 'PrimaryColor'}, inplace=True)
    df["IsMix"] = df.apply(simplifytoBoolean, args=(["Mix", "/"], "Breed"), axis=1) # Creating column that states whether animal is a mix or not
    df["Breed"] = df["Breed"].replace(" Mix", "/", regex=True) # Cutting everything after / or space in Breed column
    df["Breed"] = df["Breed"].replace("\/.*", "", regex=True)
    df.loc[df["SexuponOutcome"].isnull(),"SexuponOutcome"] = "Unknown" #fill empty cells with value "unknown"
    df.loc[df["AgeuponOutcome"].isnull(),"AgeuponOutcome"] = "Unknown" #fill empty cells with value "unknown"

    # Creating column that states whether the animal is neutered/spayed
    df["N/S"] = df.apply(simplifytoBoolean, args=(["Spayed", "Neutered"], "SexuponOutcome"), axis=1) # N/S is shortened form for Neutered/Spayed

    # Cut everything other than Male/Female from SexuponOutcome column
    df["SexuponOutcome"] = df["SexuponOutcome"].replace(".* ", "", regex=True)

    # Convert all ages to same units (days) for AgeuponOutcome column
    df["AgeInDays"] = df.apply(convertAge, axis=1)
    df = df.drop("AgeuponOutcome", axis=1) # remove DateTime column now that we've extracted useful information 

    # Convert DateTime to Month Column, DayofWeek Column, and ApproxHour Column
    df["Month"] = df.apply(convertDateTime, args=("Month",), axis=1)
    df["DayofWeek"] = df.apply(convertDateTime, args=("Day",), axis=1)
    df["ApproxHour"] = df.apply(convertDateTime, args=("Hour",), axis=1)
    df = df.drop("DateTime", axis=1) # remove DateTime column now that we've extracted useful information 

    return df

# learning rate, epochs, batch size, validation split and patience
learning_rate = 0.001
epochs = 100
batch_size = 16
validation_split = 0.2222222
patience = 4
lr_update_factor = 0.1
lr_patience = 2

# Load the Dataset
data = pd.read_csv("train_90.csv") # store csv into dataframe (data)
data = dataManipulation(data)
# load the test data
test_data = pd.read_csv("test_10.csv")
test_data = dataManipulation(test_data)

all_data = pd.concat([data, test_data])
for col in all_data.select_dtypes(include=[np.object]).columns:
    unique_values = all_data[col].dropna().unique()
    data[col] = data[col].astype(CategoricalDtype(categories=unique_values))
    test_data[col] = test_data[col].astype(CategoricalDtype(categories=unique_values))

data = categoricalToNumeric(data)
dataset = data.values

# number of input, hidden and output nodes
input_nodes = len(data.columns) - 2 
print("Input nodes {}".format(input_nodes))
output_nodes = 5 #5 possible values for output variable
hidden_nodes = int(input_nodes*.666666666 + output_nodes) #roughly 2/3 size of input layer + size of output layer:
                                                         # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

x_train = dataset[:, 1:input_nodes+1]
labels = dataset[:,len(data.columns)-1]

# encode class values as integers: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)

# convert integers to one-hot vectors
y_train = np_utils.to_categorical(encoded_labels)

model_name = 'test.h5'

# create a Keras model: LOOK  INTO ADDING MORE/DIFFERENT KINDS OF LAYERS like Conv2D, MaxPooling2D, Dropout, Flatten
model = Sequential()
model.add(Dense(hidden_nodes, activation='relu', input_shape=(input_nodes,), use_bias=False))
model.add(Dense(hidden_nodes, activation='relu', input_shape=(hidden_nodes,), use_bias=False))
model.add(Dense(output_nodes, activation='softmax', bias=False))

model.summary()
opt= optimizers.Nadam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# setup callbacks
callbacks = [
EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
ModelCheckpoint(model_name,
monitor='val_loss', save_best_only=True, verbose=1),
ReduceLROnPlateau(monitor='val_loss', factor=lr_update_factor, patience=lr_patience, verbose=1, min_delta=0.00001)]


# train the model
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=validation_split)

# save the model
model.save(model_name)
print("model saved")

saved_model_epoch_index = np.argmin(hist.history.get('val_loss', None))

print("training accuracy = {}, validation accuracy = {}".format(hist.history.get('acc')[saved_model_epoch_index], hist.history.get('val_acc')[saved_model_epoch_index]))