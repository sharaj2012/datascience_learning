#%%
import matplotlib.pyplot as plt
import numpy as np

# %%
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
# %%
# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
# %%
# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)
# %%
# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 1].plot(x, y)
axs[1, 1].scatter(x, y)
# %%
plt.subplots(2, 2, sharex='col')
# %%
