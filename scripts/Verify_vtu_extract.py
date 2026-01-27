import numpy as np 

data = np.load('fem_all_timesteps.npz')

# print(type(data["coords"]))

coords = data["coords"]
temperature = data["temperature"]
time = data["time"]
n_timesteps = data["n_timesteps"]

print(coords.shape)
print(temperature.shape)
print(time.shape)
print(n_timesteps.shape)

print(n_timesteps)