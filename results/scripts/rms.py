import numpy as np


stds = [176, 246, 225, 371]  # Known standard deviations (example)

sd=np.sqrt(np.mean(np.square(stds)))
print(sd)
