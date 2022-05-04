#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[43]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
import statistics

def comparetonormal(array=[], name = ""):

    x_axis = np.arange(min(array), max(array), (max(array)-min(array))/100)
    mean = np.mean(x_axis)
    sd = np.var(x_axis)**(1/2)

    plt.plot(x_axis, max(array)*norm.pdf(x_axis, mean, sd))
    plt.hist(array, label = name)
    plt.legend()
    plt.show()
    
    
comparetonormal(zcmb_array, 'zcmb_array')
comparetonormal(zhel_array, 'zhel_array')
# comparetonormal(dz_array, 'dz_array') I think this is all zeros
comparetonormal(mb_array, 'mb_array')
comparetonormal(mbz_array, 'mbz_array')


# In[ ]:




