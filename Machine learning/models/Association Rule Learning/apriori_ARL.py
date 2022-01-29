# Apriori

##Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) #header = none is for specifying the machine that the ther is no title in the database
transactions = [] # initialising an empty list
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
# The above lines of code are to convert the array into a suitable datatypy for apriori that is a list of lists
    
# Training the apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# All the parameters of the apriori class are determined by hit and trial method and varies as per the requirement

# Visualising the results
results = list(rules)