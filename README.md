# Flight_Chanrge_Prediction
Predicting the flight prices


This is a online competation where we are asked to predict the flight price.

## Data:
Data sets which are provided are:
### Data_Train.csv
### Data_Test.csv

## Preparing the data on excel:
I used the data sets given by them using excel and transforming the data into 
### DATA_TRAIN - Cop.csv
### DATA_TEST - Copy.csv

Using Excel I extracted the intermediate stops as seperate columns, extracting the time and date seperately from the arrival details.

## Python file:

Flight price prediction.ipynb:

Using these data sets, Flight price prediction.ipynb has the procedure of extracting few more features like Day of the week of travel, time of travel, weather it is a early morning flight or not.
I used Random Forest with simple parameters at first then I tried using GBM followed by using grid search with random forest then NN also (ANN.py).

With these predictions I was able t get a 0.941 accuracy. 
