from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import matplotlib.pyplot as plt
import time
from brainflow.data_filter import WindowOperations
debug = 1
params = BrainFlowInputParams()
params.serial_port = 'COM6' #Change this depending on your device and OS
board_id = 39 #Change this depending on your device
threshold = 0.8*0.265608014

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
#Releases the board session
board.release_session()

board.prepare_session()
print("Starting Stream")
board.start_stream()
for i in range(200):
    time.sleep(0.8)
    current_data = board.get_current_board_data(200)
    print(i)
    psd = DataFilter.get_psd(current_data.flatten(), 256, WindowOperations.HANNING)
    theta_power = DataFilter.get_band_power(psd, 4, 8)
    alpha_power = DataFilter.get_band_power(psd, 8, 13)
    beta_power = DataFilter.get_band_power(psd, 13, 32)
    print(theta_power)
    
    # concentration = concentration = beta_power / (alpha_power + theta_power)
    # concentration = (theta_power + beta_power) / alpha_power/2
    # alertness = beta_power/alpha_power
    # attention = theta_power/alpha_power


    # if concentration < threshold:
    #     print("Alert: Concentration level has decreased.")
    # else:
    #     print("Concentration level is within normal range.")

data = board.get_board_data()
print(data.shape)
print("Ending stream")
board.stop_stream()
board.release_session()

eeg_channels = board.get_eeg_channels(board_id)
eeg_data = data[eeg_channels]
plt.plot(np.arange(eeg_data.shape[1]), eeg_data[0])

if debug == 1:
    print(eeg_data.shape)
    DataFilter.write_file(eeg_data, 'eeg_data_test.csv', 'w') #Writes into a csv file in the current directory

    restored_data = DataFilter.read_file('eeg_data_test.csv') #Reads file back
    print(restored_data.shape)

    #This shows how much the saved data differs from the original data, they are very similar but not equal.
    print(eeg_data - restored_data)

for channel in range(eeg_data.shape[0]):
    #Filters work in place
    DataFilter.perform_lowpass(eeg_data[channel], BoardShim.get_sampling_rate(board_id), 50.0, 5,
                                       FilterTypes.BUTTERWORTH, 1)
    DataFilter.perform_highpass(eeg_data[channel], BoardShim.get_sampling_rate(board_id), 2.0, 4,
                                        FilterTypes.BUTTERWORTH, 0)
plt.plot(np.arange(eeg_data.shape[1]), eeg_data[0])


