
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import matplotlib.pyplot as plt
import time
from brainflow.data_filter import WindowOperations


params = BrainFlowInputParams()
params.serial_port = 'COM6' #Change this depending on your device and OS
board_id = 39 #Change this depending on your device
beta=[]
theta=[]


#Prepares the board for reading data
try:
    board_id = 39
    board = BoardShim(board_id, params)
    board.prepare_session()
    print("Successfully prepared physical board.")
except Exception as e:
    print(e)
    #If the device cannot be found or is being used elsewhere, creates a synthetic board instead
    print("Device could not be found or is being used by another program, creating synthetic board.")
    board_id = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(board_id, params)
    board.prepare_session()


print("Starting Stream")
board.start_stream()
eeg_channels = board.get_eeg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
time.sleep(2)

for i in range(200):
    time.sleep(0.5)
    current_data = board.get_current_board_data(128)
    current_eeg_data = current_data[eeg_channels] 
    theta_total, alpha_total, beta_total = 0, 0, 0

    for channel_data in current_eeg_data:
        if len(channel_data) > 0:

        # Apply filters to clean the signal
            DataFilter.perform_lowpass(channel_data, BoardShim.get_sampling_rate(board_id), 40.0, 5,
                                    FilterTypes.CHEBYSHEV_TYPE_1, 0.5)
            DataFilter.perform_highpass(channel_data, BoardShim.get_sampling_rate(board_id), 2.0, 4,
                                        FilterTypes.CHEBYSHEV_TYPE_1, 0.5)
        # Compute PSD
            psd = DataFilter.get_psd(channel_data, sampling_rate, WindowOperations.HANNING)
            #print(psd)
        # Extract band powers
            theta_power = DataFilter.get_band_power(psd, 4, 8)
            beta_power = DataFilter.get_band_power(psd, 13, 32)

            theta_total += theta_power
            beta_total += beta_power
    

    print(i)
    print(beta_total,theta_total)
    beta.append(beta_total)
    theta.append(theta_total)

data = board.get_board_data()
print(data.shape)
print("Ending stream")
board.stop_stream()
board.release_session()

print(beta)
print("Concentrate:", beta[0:49], (sum(beta[0:49])/50))
print("Yap:", beta[50:99], (sum(beta[50:99])/50))
print("Look away:", beta[100:149], (sum(beta[100:149])/50))
print("Zoning:", beta[150:199], (sum(beta[150:199])/50))

print(theta)
print("Awake:", theta[0:49], (sum(theta[0:49])/50))
print("closed:", theta[50:99], (sum(theta[50:99])/50))
print("head tilt:", theta[100:149], (sum(theta[100:149])/50))
print("Awake:", theta[150:199], (sum(theta[150:199])/50))


eeg_channels = board.get_eeg_channels(board_id)
eeg_data = data[eeg_channels]
plt.plot(np.arange(eeg_data.shape[1]), eeg_data[0])


print(eeg_data.shape)
DataFilter.write_file(eeg_data, 'eeg_data_test.csv', 'w') #Writes into a csv file in the current directory

restored_data = DataFilter.read_file('eeg_data_test.csv') #Reads file back
print(restored_data.shape)