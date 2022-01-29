# Upper Confidence Bound Reinforced Learning model

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Problem Statement - In order to make the best strategy to place the ads as per their conversion rate according to the database provided by the company
#Import the Dataset 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000  # Number of users
d = 10  # Number of ads
ads_selected = []
numbers_of_selections = [0] * d #The number of times the ad i was selected
Sums_of_rewards = [0] * d # The su of rewards of the ad i
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = Sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    Sums_of_rewards[ad] = Sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualising the results
plt.hist(ads_selected)
plt.title("Histogram for the ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times the ad was selected")
plt.show()
