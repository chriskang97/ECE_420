import numpy as np
import matplotlib.pyplot as plt
import os
def peak_detection(t, sig, thresh):
    peaks = []
    max_val = -np.Inf
    prev_value = -np.Inf
    N = len(sig)

    for i in range(0, N-1):

        next_value = sig[i+1]

        if sig[i] > max_val and sig[i] > thresh:
            max_val = sig[i]
            position = t[i]

        if sig[i] < 0 and max_val != -np.Inf :
            peaks.append((position, max_val))
            max_val = -np.Inf


    return np.array(peaks)

os.chdir('C:/Users/chris/PycharmProjects/lab1')

csv_filename = 'sample_sensor_data.csv'
data = np.genfromtxt(csv_filename, delimiter=',').T
timestamps = (data[0] - data[0,0]) / 1000

accel_data = data[1:4]
gyro_data = data[4:-1]
max_peaks = peak_detection(timestamps, accel_data[0], 3)


plt.plot(timestamps, accel_data[0])
plt.title("First axis of accelerometer data")
plt.xlabel("Time")
plt.ylabel("Meters per second")
plt.scatter(max_peaks[:, 0], max_peaks[:, 1], color='red')
plt.show()


