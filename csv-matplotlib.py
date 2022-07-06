from turtle import color
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('/home/bimanjaya/learner/TA/brainflow/data_logger/2022-07-05.csv')

# for column in df.columns:
#     print(df[column])
#     plt.figure()
#     plt.title(column)
#     plt.plot(df[column])
#     plt.show()

plt.figure()
plt.title('Graph')
plt.plot(df['Channel1'],color='r')
plt.plot(df['Channel2'],color='g')
plt.plot(df['Channel3'],color='b')
plt.plot(df['Channel4'],color='y')
plt.show()