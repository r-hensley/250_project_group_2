#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

with open('lcparam_DS17f.txt', 'r') as f:
    data = f.readlines()  # a list of the lines in the file
    data = data[1:]  # ignore the first line in the file which is just column titles
    data = [' '.join(i.split()[:6]) for i in data]  # remove unused data columns
name_array = np.array([float(i.split()[0]) for i in data])  # I don't think we need to use this

# Define arrays for each parameter from the data variable
zcmb_array = np.array([float(i.split()[1]) for i in data])
zhel_array = np.array([float(i.split()[2]) for i in data])
dz_array = np.array([float(i.split()[3]) for i in data])  # these are all zero
mb_array = np.array([float(i.split()[4]) for i in data])
mbz_array = np.array([float(i.split()[5]) for i in data])

# Display samples of the data and defined variables
print(f"Example first data event: {data[0]}")
print(f"name_array: {', '.join(list(map(str, name_array[:4])))}, ...")
print(f"zcmb_array: {', '.join(list(map(str, zcmb_array[:4])))}, ...")
print(f"zhel_array: {', '.join(list(map(str, zhel_array[:4])))}, ...")
print(f"dz_array: {', '.join(list(map(str, dz_array[:4])))}, ...")
print(f"mb_array: {', '.join(list(map(str, mb_array[:4])))}, ...")
print(f"mbz_array: {', '.join(list(map(str, mbz_array[:4])))}, ...")


# In[4]:


covariance = np.cov([zcmb_array, zhel_array, dz_array, mb_array, mbz_array])

print(covariance)


# In[5]:


# Let's test the likelihood of the first element of the zcmb_array as an example

def likelihood(array = [], point=0, variance = 0):
    l = np.mean(array)
    #  This might be different depending on which variables we are looking at
    likelihood = (1/(np.sqrt(2*np.pi)*variance))*np.exp(-1*((point-l)**2)/(2*variance**2))     # Likelihood using a Gaussian
    return likelihood

# I suspect the variance used here is incorrect. I believe we pull it out of the covariance array somehow
print("Likelihood of first element of zmbc_array:", likelihood(zcmb_array, zcmb_array[0], np.var(zcmb_array)))  



# In[ ]:




