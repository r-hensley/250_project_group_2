import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

with open('../data/lcparam_DS17f.txt', 'r') as f:
    data = f.readlines()  # a list of the lines in the file
    data = data[1:]  # ignore the first line in the file which is just column titles
    data = [' '.join(i.split()[:6]) for i in data]  # remove unused data columns

# Define arrays for each parameter from the data variable
zcmb_array = np.array([float(i.split()[1]) for i in data])
zhel_array = np.array([float(i.split()[2]) for i in data])
mb_array = np.array([float(i.split()[4]) for i in data])
mbz_array = np.array([float(i.split()[5]) for i in data])


def comparetonormal(array=None, name=""):
    if array is None:
        array = []  # Default argument of a function should not be a mutable type

    # Generate 100 values equally spaced between min and max of array
    x_axis = np.linspace(min(array), max(array), 100)
    mean = np.mean(x_axis)
    sd = np.var(x_axis)**(1/2)

    plt.plot(x_axis, max(array) * norm.pdf(x_axis, mean, sd))
    plt.hist(array, label=name)
    plt.legend()

    plt.savefig(f"./compare_to_gaussian_images/{name}.png")
    plt.show()


comparetonormal(zcmb_array, 'zcmb_array')
comparetonormal(zhel_array, 'zhel_array')
# comparetonormal(dz_array, 'dz_array') This is all zeros
comparetonormal(mb_array, 'mb_array')
comparetonormal(mbz_array, 'mbz_array')
