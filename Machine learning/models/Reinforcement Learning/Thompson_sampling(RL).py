# Thompson Sampling 

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Import the Dataset 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thonpson sampling on the dataset
N = 10000
d = 10
ads_selected = []
Numbers_of_rewards_1 = [0] * d 
Numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0  
    for i in range(0, d):
        random_beta = random.betavariate(Numbers_of_rewards_1[i] + 1, Numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        Numbers_of_rewards_1[ad] = Numbers_of_rewards_1[ad] + 1
    else:
        Numbers_of_rewards_0[ad] = Numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

#Visualising the results
plt.hist(ads_selected)
plt.title("Histogram for the ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times the ad was selected")
plt.show()

