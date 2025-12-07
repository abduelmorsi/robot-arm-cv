import matplotlib.pyplot as plt
import numpy as np
from bitalino import BITalino
import time

# === CONFIGURATION ===
# mac_address =   # Replace with your BITalino MAC address
sampling_rate = 1000              # Hz
channels = [0]                    # Read EMG from channel A0
buffer_size = 100                 # Frames per read
plot_window = 500                 # Number of samples shown on plot

# === INITIALIZATION ===
device = BITalino("COM4")  # Directly pass MAC address (newer API)
print("Connected to BITalino")

# Optional battery threshold setup (0–63 → ~3.4V–3.8V)
device.battery(20)


# Get firmware version
version = device.version()
print("Firmware version:", version)

# === START ACQUISITION ===
device.start(sampling_rate, channels)
print("Started acquisition on channels:", channels)

# === PLOTTING SETUP ===
plt.ion()
fig, ax = plt.subplots()
ydata = np.zeros(plot_window)
xdata = np.arange(plot_window)
line, = ax.plot(xdata, ydata)
ax.set_ylim(0, 1023)
ax.set_xlim(0, plot_window)
ax.set_title("Live EMG Plot (Channel A0)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Signal Level")

try:
    while True:
        data = device.read(buffer_size)
        analog_data = data[:, 5]  # A0 is at index 5 in the frame

        # Update rolling data
        ydata = np.concatenate((ydata[len(analog_data):], analog_data))
        line.set_ydata(ydata)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("User interrupted.")

finally:
    # === CLEANUP ===
    device.stop()
    device.close()
    plt.ioff()
    plt.show()
    print("Connection closed.")
