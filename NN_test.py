#Keras model with the same architecture as that in "Make You Own Neural Network" by T. Rashid
import tensorflow as tf
import numpy as np
#to measure elapsed time
from timeit import default_timer as timer
import datetime
#pandas for reading CSV files
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys
import csv
import heapq

#np.set_printoptions(threshold=sys.maxsize)
#setting for dynamic memory allocation in GPU to avoid CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def simplifytoBoolean(row, strings, col):
    if any(x in row[col] for x in strings):
        boolVal = "Yes"
        val = 1
    elif row[col] == "Unknown":
        boolVal = "Unknown"
        val = -1
    else:
        boolVal = "No"
        val = 0
    return val

def secondaryColor(row):
    identifier = "/"
    if identifier in row["Color"]:
        secondaryColor = row["Color"].split(identifier,1)[1]
    else:
        secondaryColor = "None"
    return secondaryColor

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

def convertDateTime(row, conversion):
    cell = row["DateTime"]
    date = datetime.datetime.strptime(cell, "%m/%d/%Y %H:%M")

    if conversion == "Month":
        convertedUnit = date.month
    elif conversion == "Day":
        convertedUnit = date.strftime('%A')
    elif conversion == "Hour":
        convertedUnit = date.hour
    else:
        print("Unexpected conversion in convertDateTime function")
        exit(1)

    return convertedUnit

def categoricalToNumeric(df):
    #df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "SecondaryColor", "IsMix", "N/S", "DayofWeek"])
    df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "IsMix", "N/S", "DayofWeek"])

    #Re-arrange dataframe to put OutcomeType as the last column
    if 'OutcomeType' in df.columns:
        cols = list(df)
        cols.insert(len(cols), cols.pop(cols.index('OutcomeType')))
        df = df.ix[:, cols]
    return df

def dataManipulation(df):
    #df = df.drop("AnimalID", axis=1) 
    df = df.drop("Name", axis=1) #don't think name or outcomesubtype are relevant, so removing from dataframe for now
    if 'OutcomeSubtype' in df.columns:
        df = df.drop("OutcomeSubtype", axis=1)
    if 'AnimalID' in df.columns:
        df.rename(columns={'AnimalID' : 'ID'}, inplace=True)

    #df["SecondaryColor"] = df.apply(secondaryColor, axis=1) #creating column with the secondary color of the animal

    num1 = df["Color"].nunique() #just to see how many unique values
    df["Color"] = df["Color"].replace("\/.*", "", regex=True) #Cutting everything after / in Color column
    df.rename(columns={'Color' : 'PrimaryColor'}, inplace=True)
    num2 = df["PrimaryColor"].nunique()
    df["IsMix"] = df.apply(simplifytoBoolean, args=(["Mix", "/"], "Breed"), axis=1) #Creating column that states whether animal is a mix or not

    num3 = df["Breed"].nunique()

    df["Breed"] = df["Breed"].replace(" Mix", "/", regex=True) #Cutting everything after / or space in Breed column
    df["Breed"] = df["Breed"].replace("\/.*", "", regex=True)

    num4 = df["Breed"].nunique()

    df.loc[df["SexuponOutcome"].isnull(),"SexuponOutcome"] = "Unknown" #fill empty cells with value "unknown"
    df.loc[df["AgeuponOutcome"].isnull(),"AgeuponOutcome"] = "Unknown" #fill empty cells with value "unknown"

    #Creating column that states whether the animal is neutered/spayed
    df["N/S"] = df.apply(simplifytoBoolean, args=(["Spayed", "Neutered"], "SexuponOutcome"), axis=1) #N/S is shortened form for Neutered/Spayed

    #Cut everything other than Male/Female from SexuponOutcome column
    df["SexuponOutcome"] = df["SexuponOutcome"].replace(".* ", "", regex=True)

    #Convert all ages to same units (days) for AgeuponOutcome column
    df["AgeInDays"] = df.apply(convertAge, axis=1)
    df = df.drop("AgeuponOutcome", axis=1) #remove DateTime column now that we've extracted useful information 

    #Convert DateTime to Month Column, DayofWeek Column, and ApproxHour Column
    df["Month"] = df.apply(convertDateTime, args=("Month",), axis=1)
    df["DayofWeek"] = df.apply(convertDateTime, args=("Day",), axis=1)
    df["ApproxHour"] = df.apply(convertDateTime, args=("Hour",), axis=1)
    df = df.drop("DateTime", axis=1) #remove DateTime column now that we've extracted useful information 

    return df
data = pd.read_csv("train_90.csv") # store csv into dataframe (data)
data = dataManipulation(data)

test_data = pd.read_csv("test_10.csv")
test_data = dataManipulation(test_data)
test_data["OutcomeType"] = 'None'
all_data = pd.concat([data, test_data])

#to ensure that training and test data have same number of inputs, due to different breeds, colors, etc. being present in one set and not the other
for col in all_data.select_dtypes(include=[np.object]).columns:
    unique_values = all_data[col].dropna().unique()
    data[col] = data[col].astype(CategoricalDtype(categories=unique_values))
    test_data[col] = test_data[col].astype(CategoricalDtype(categories=unique_values))

data = categoricalToNumeric(data)
test_data = categoricalToNumeric(test_data)
model = load_model('final_model.h5')
input_nodes = len(data.columns) - 2
outputs_array = np.zeros((len(test_data.index), 5))

for index, row in test_data.iterrows():
    inputs = (row[1:input_nodes+1]).values
    outputs = model.predict(np.reshape(inputs, (1, len(inputs)))) #numpy ndarray type
    outputs_array[index,:] = outputs     

results = pd.DataFrame(outputs_array, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
results.insert(0, "ID", test_data["ID"])

results.to_csv(r'results.csv', index=None)