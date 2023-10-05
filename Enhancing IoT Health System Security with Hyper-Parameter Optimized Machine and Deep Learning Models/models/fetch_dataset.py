# -*- coding: utf-8 -*-
"""Fetch Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z_9SD8H9CHZsJFFZMojatBicpXZa08fZ
"""

"""
   Obtain the Dataset from its website:
"""
# Obtain the Train-Test Dataset:
public_file = cloudstor(url="https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i", password='')

#Establish that this is a directory (not a file):
print(public_file.remote_type)

#Query the directory and retrieve a (Python) list of the files and subdirectories:
ls_entries = public_file.list()
print(ls_entries)
print(public_file.is_file(ls_entries[0]))

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_Weather:
IoT_Weather = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Weather.csv", "IoT_Weather.csv")
data = open('IoT_Weather.csv', newline='', encoding='utf-8')
IoT_Weather = csv.DictReader(data)
# read dataframe
IoT_Weather = pd.DataFrame(IoT_Weather)

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_Thermostat:
IoT_Thermostat = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Thermostat.csv", "IoT_Thermostat.csv")
data = open('IoT_Thermostat.csv', newline='', encoding='utf-8')
IoT_Thermostat = csv.DictReader(data)
# read dataframe
IoT_Thermostat = pd.DataFrame(IoT_Thermostat)

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_Motion_Light:
IoT_Motion_Light = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Motion_Light.csv", "IoT_Motion_Light.csv")
data = open('IoT_Motion_Light.csv', newline='', encoding='utf-8')
IoT_Motion_Light = csv.DictReader(data)
# read dataframe
IoT_Motion_Light = pd.DataFrame(IoT_Motion_Light)

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_Modbus:
IoT_Modbus = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Modbus.csv", "IoT_Modbus.csv")
data = open('IoT_Modbus.csv', newline='', encoding='utf-8')
IoT_Modbus = csv.DictReader(data)
# read dataframe
IoT_Modbus = pd.DataFrame(IoT_Modbus)

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_GPS_Tracker:
IoT_GPS_Tracker = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_GPS_Tracker.csv", "IoT_GPS_Tracker.csv")
data = open('IoT_GPS_Tracker.csv', newline='', encoding='utf-8')
IoT_GPS_Tracker = csv.DictReader(data)
# read dataframe
IoT_GPS_Tracker = pd.DataFrame(IoT_GPS_Tracker)

# Obtain Train_Test_IoT_dataset/Train_Test_IoT_Fridge:
IoT_Fridge = public_file.download_file("Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv", "IoT_Fridge.csv")
data = open('IoT_Fridge.csv', newline='', encoding='utf-8')
IoT_Fridge = csv.DictReader(data)
# read dataframe
IoT_Fridge = pd.DataFrame(IoT_Fridge)