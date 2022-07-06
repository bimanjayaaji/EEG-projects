# https://github.com/openbci-archive/pyOpenBCI/blob/master/Examples/print_raw_example.py

from pyOpenBCI import OpenBCIGanglion
import matplotlib.pyplot as plt 
import numpy as np
import datetime
import time
import csv

def date_now():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today = str(today)
    return today

def time_now():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    return str(now)

def start_writer():
    file = open('/home/bimanjaya/learner/TA/brainflow/data_logger/'+date_now()+'.csv', 'w', newline ='')
    with file:
        header = ['Channel1','Channel2','Channel3','Channel4']
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
    file.close()

def print_raw(sample):
    file = open('/home/bimanjaya/learner/TA/brainflow/data_logger/'+date_now()+'.csv', 'a+', newline ='')
    if time.time() - start_time <= 10:
        print(sample.channels_data)
        sample_list = sample.channels_data.tolist()
        sample_list = [sample_list]

        plt.plot(sample.channels_data[0],color='r')
        plt.plot(sample.channels_data[1],color='g')
        plt.plot(sample.channels_data[2],color='b')
        plt.plot(sample.channels_data[3],color='y')
        plt.pause(0.05)

        with file:   
            write = csv.writer(file)
            write.writerows(sample_list)
        print(round(time.time())-start_time)
    else:
        board.stop_stream()
        file.close()

#Set (daisy = True) to stream 16 ch 
board = OpenBCIGanglion(mac='D6:B8:88:C4:46:70')

if __name__ == '__main__':
    start_writer()

    plt.figure()
    plt.title('Graph')
    start_time = time.time()
    board.start_stream(print_raw)
    plt.show()



