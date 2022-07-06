import matplotlib.pyplot as plt
import scipy.fftpack
import pandas as pd
import numpy as np
import scipy as sp

df = pd.read_csv('/home/bimanjaya/learner/TA/brainflow/EEG-projects/data_logger/2022-07-06.csv')

channel1 = df['Channel2']

eeg_fft = sp.fftpack.fft(channel1)

xf = sp.fftpack.fftfreq(len(channel1), 1 / len(channel1))

plt.plot(xf, np.abs(eeg_fft))
plt.show()

'''
------------------------------------------------------------
'''




