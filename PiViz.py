import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
from collections import deque
import subprocess
import threading
import queue
import time

class AudioVisualizer:
    def __init__(self):
        """
        Initialize the AudioVisualizer class with default settings.
        """
        # Audio settings
        self.sample_rate = None  # Will be set dynamically based on microphone's default sample rate
        self.FFT_WINDOW_SIZE = None  # Will be determined based on sample rate
        self.frequencies = None  # Frequency bins for plotting FFT

        # Global Variables for Consistent Axis Associations
        self.MIN_FREQ = 0      # Minimum frequency in Hz
        self.MAX_FREQ = 20000  # Maximum frequency in Hz (can be adjusted by user)
        self.MIN_AMP = 0       # Minimum amplitude for FFT plot
        self.MAX_AMP = 10      # Maximum amplitude for FFT plot
        self.TIME_WINDOW = 5   # Time window in seconds for waveform and spectrogram (default is 5 seconds)

        # Data structures for audio and visualization
        self.waveform_data = None  # Stores audio samples for waveform
        self.waveform_time = None  # Time stamps for waveform data
        self.line_waveform = None  # Matplotlib line object for waveform
        self.line_fft = None  # Matplotlib line object for FFT
        self.img_spectrogram = None  # Matplotlib image object for spectrogram
        self.time_data = None  # Time stamps for spectrogram
        self.spectrogram_data = None  # FFT data for spectrogram

        # Visualization flags
        self.show_waveform = False  # Whether to display waveform plot
        self.show_fft = False  # Whether to display FFT plot
        self.show_spectrogram = False  # Whether to display spectrogram

        # Queues for thread-safe communication between audio callback and main thread
        self.audio_queue = queue.Queue()  # For waveform data
        self.fft_queue = queue.Queue()  # For FFT data
        self.spectrogram_queue = queue.Queue()  # For spectrogram data

        # Control flag for main loop
        self.running = True  # Set to False to stop the main loop

        # Sample counter for generating continuous timestamps
        self.sample_counter = 0  # Initialize sample counter

    def stream_and_save_srt(input_srt, output_file):
        """
        Stream SRT audio while simultaneously saving to a file.

        Parameters:
        - input_srt (str): The SRT stream URL (e.g., 'srt://192.168.1.157:5000?mode=caller').
        - output_file (str): Path to the file where audio will be saved.
        """
        ffmpeg_command = [
            "ffmpeg",
            "-i", input_srt,
            "-map", "0:a", "-c:a", "copy", output_file,  # Save to file
            "-map", "0:a", "-f", "wav", "-"  # Stream to display
        ]

        try:
            # Start the FFmpeg process
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Streaming and saving started. Press Ctrl+C to stop.")
            process.wait()  # Wait for the process to complete
        except KeyboardInterrupt:
            print("Stopping stream...")
            process.terminate()
            process.wait()

    def setup_plots(self):
        """
        Set up the Matplotlib plots for the selected visualizations.
        """
        # Enable interactive plotting mode
        plt.ion()

        # Determine the number of subplots needed
        num_plots = sum([self.show_waveform, self.show_fft, self.show_spectrogram])
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        # Ensure axes is iterable
        if num_plots == 1:
            axes = [axes]

        self.fig = fig  # Reference to the figure object
        plot_index = 0  # Index for accessing axes

        # Set up waveform plot if selected
        if self.show_waveform:
            self.ax_waveform = axes[plot_index]
            # Initialize an empty line plot
            self.line_waveform, = self.ax_waveform.plot([], [], lw=1.5)
            # Set plot labels and title
            self.ax_waveform.set_xlabel('Time (s)')
            self.ax_waveform.set_ylabel('Amplitude')
            self.ax_waveform.set_title("Real-Time Waveform")
            # Set y-axis limits to match audio data range
            self.ax_waveform.set_ylim(-1.0, 1.0)
            plot_index += 1  # Move to next subplot

        # Set up FFT plot if selected
        if self.show_fft:
            self.ax_fft = axes[plot_index]
            # Initialize an empty line plot
            self.line_fft, = self.ax_fft.plot([], [], lw=1.5)
            # Set plot labels and title
            self.ax_fft.set_xlabel('Frequency (Hz)')
            self.ax_fft.set_ylabel('Amplitude')
            self.ax_fft.set_title("Real-Time FFT Visualization")
            # Set x-axis limits to frequency range
            self.ax_fft.set_xlim(self.MIN_FREQ, self.MAX_FREQ)
            # Set y-axis limits to amplitude range
            self.ax_fft.set_ylim(self.MIN_AMP, self.MAX_AMP)
            plot_index += 1

        # Set up spectrogram plot if selected
        if self.show_spectrogram:
            self.ax_spectrogram = axes[plot_index]
            # Initialize an empty image with higher contrast colormap
            self.img_spectrogram = self.ax_spectrogram.imshow(
                np.zeros((100, 512)),  # Placeholder data
                aspect='auto',
                origin='lower',
                cmap='plasma',  # Higher contrast colormap
                extent=[0, 1, self.MIN_FREQ, self.MAX_FREQ],  # Will be updated dynamically
                vmin=-20,
                vmax=0
            )
            # Set plot labels and title
            self.ax_spectrogram.set_xlabel("Time (s)")
            self.ax_spectrogram.set_ylabel("Frequency (Hz)")
            self.ax_spectrogram.set_title("Spectrogram")
            # Set y-axis limits to frequency range
            self.ax_spectrogram.set_ylim(self.MIN_FREQ, self.MAX_FREQ)
            plot_index += 1

        # Adjust the layout to prevent overlap
        self.fig.tight_layout()
        # Draw the initial plots
        self.fig.canvas.draw_idle()

    def update_waveform_plot(self):
        """
        Update the waveform plot with the latest audio data.
        """
        try:
            # Get all available audio data from the queue
            while True:
                audio_data, timestamps = self.audio_queue.get_nowait()
                self.waveform_data.extend(audio_data)
                self.waveform_time.extend(timestamps)
        except queue.Empty:
            # No more data to process
            pass

        if self.waveform_data:
            # Use data within the sliding time window
            current_time = self.waveform_time[-1]
            min_time = current_time - self.TIME_WINDOW
            # Find indices within the time window
            indices = np.where((self.waveform_time >= min_time))[0]
            data = np.array(self.waveform_data)[indices]
            time_axis = np.array(self.waveform_time)[indices]

            # Update the waveform plot
            self.line_waveform.set_data(time_axis, data)
            self.ax_waveform.set_xlim(min_time, current_time)
            self.ax_waveform.set_ylim(-1.0, 1.0)

    def update_fft_plot(self):
        """
        Update the FFT plot with the latest frequency data.
        """
        try:
            # Get the latest FFT magnitude data from the queue
            y_magnitude = self.fft_queue.get_nowait()

            # Limit frequencies to the selected range
            freq_indices = np.where((self.frequencies >= self.MIN_FREQ) & (self.frequencies <= self.MAX_FREQ))
            frequencies = self.frequencies[freq_indices]
            y_magnitude = y_magnitude[freq_indices]

            # Normalize amplitude to be between MIN_AMP and MAX_AMP
            max_magnitude = np.max(y_magnitude)
            if max_magnitude == 0:
                max_magnitude = 1e-6  # Avoid division by zero
            y_magnitude_normalized = y_magnitude / max_magnitude * self.MAX_AMP

            # Update the FFT plot
            self.line_fft.set_data(frequencies, y_magnitude_normalized)
            self.ax_fft.set_xlim(self.MIN_FREQ, self.MAX_FREQ)
            self.ax_fft.set_ylim(self.MIN_AMP, self.MAX_AMP)
        except queue.Empty:
            # No new FFT data available
            pass

    def update_spectrogram(self):
        """
        Update the spectrogram with the latest FFT data.
        """
        try:
            # Get the latest FFT data for spectrogram
            fft_data, current_time = self.spectrogram_queue.get_nowait()
            self.spectrogram_data.append(fft_data)
            self.time_data.append(current_time)
        except queue.Empty:
            # No new spectrogram data available
            pass

        if len(self.spectrogram_data) > 1:
            # Convert spectrogram data to numpy array
            spec_data = np.array(self.spectrogram_data).T

            # Convert amplitude to decibel scale
            spec_data_db = 10 * np.log10(spec_data + 1e-6)

            # Clip data for better visualization contrast
            spec_data_db = np.clip(spec_data_db, a_min=-20, a_max=0)

            # Limit frequencies to the selected range
            freq_bins = self.frequencies
            freq_indices = np.where((freq_bins >= self.MIN_FREQ) & (freq_bins <= self.MAX_FREQ))
            freq_bins = freq_bins[freq_indices]
            spec_data_db = spec_data_db[freq_indices]

            # Limit time data to a sliding 5-second window
            current_time = self.time_data[-1]
            min_time = current_time - 5  # 5 seconds window
            time_indices = np.where((np.array(self.time_data) >= min_time))[0]
            time_axis = np.array(self.time_data)[time_indices]
            spec_data_db = spec_data_db[:, time_indices]

            # Update the spectrogram image
            self.img_spectrogram.set_data(spec_data_db)
            self.img_spectrogram.set_extent([
                min_time,  # Start time
                current_time,  # End time
                self.MIN_FREQ,  # Min frequency
                self.MAX_FREQ  # Max frequency
            ])
            self.img_spectrogram.set_cmap('inferno')  # Custom colormap: Black, Red, Yellow
            self.ax_spectrogram.set_xlim(min_time, current_time)
            self.ax_spectrogram.set_ylim(self.MIN_FREQ, self.MAX_FREQ)

            # Redraw the canvas
            self.fig.canvas.draw_idle()

    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function for processing incoming audio data.
        """
        if status:
            # Print any stream status messages
            print("Stream status:", status)

        # Make a copy of the audio data
        audio_data = indata[:, 0].copy()

        # Save current sample counter
        current_sample_counter = self.sample_counter

        # Generate continuous timestamps for the audio data
        timestamps = (np.arange(len(audio_data)) + current_sample_counter) / self.sample_rate

        # Increment sample counter
        self.sample_counter += len(audio_data)

        # Put the audio data and timestamps into the waveform queue
        if self.show_waveform:
            self.audio_queue.put((audio_data, timestamps))

        # Process FFT and spectrogram data
        if self.show_fft or self.show_spectrogram:
            # Ensure audio data length matches FFT window size
            if len(audio_data) < self.FFT_WINDOW_SIZE:
                # Pad zeros to match FFT window size
                audio_data_padded = np.pad(
                    audio_data,
                    (0, self.FFT_WINDOW_SIZE - len(audio_data)),
                    'constant'
                )
                # Since we padded zeros, adjust the sample counter accordingly
                fft_sample_counter = current_sample_counter + len(audio_data_padded) // 2
            else:
                audio_data_padded = audio_data[:self.FFT_WINDOW_SIZE]
                fft_sample_counter = current_sample_counter + len(audio_data_padded) // 2

            # Apply a window function to reduce spectral leakage
            window = np.hanning(len(audio_data_padded))
            audio_windowed = audio_data_padded * window

            # Compute the one-sided FFT (positive frequencies only)
            yf = np.abs(np.fft.rfft(audio_windowed))
            # Compute frequencies if not already done
            if self.frequencies is None:
                self.frequencies = np.fft.rfftfreq(self.FFT_WINDOW_SIZE, d=1 / self.sample_rate)

            # Current time for spectrogram
            fft_time = fft_sample_counter / self.sample_rate

            # Put the FFT magnitude data into the FFT queue
            if self.show_fft:
                self.fft_queue.put(yf)

            # Put the spectrogram data into the spectrogram queue
            if self.show_spectrogram:
                self.spectrogram_queue.put((yf, fft_time))

    def start_live_audio_stream(self):
        """
        Start streaming live audio from the microphone.
        """
        # Dynamically determine the sample rate
        if self.sample_rate is None:
            self.sample_rate = int(sd.query_devices(sd.default.device[0], 'input')['default_samplerate'])
            print(f"Detected sample rate: {self.sample_rate} Hz")

        # Adjust FFT_WINDOW_SIZE for desired resolution
        self.FFT_WINDOW_SIZE = int(self.sample_rate / 10)  # Adjust as needed

        # Initialize buffers for waveform and spectrogram data
        #maxlen = int(self.buffer_duration_seconds * self.sample_rate)
        #self.waveform_data = deque(maxlen=maxlen)
        #self.waveform_time = deque(maxlen=maxlen)
        self.waveform_data = deque(maxlen=44100 * 5)  # 5 seconds
        self.waveform_time = deque(maxlen=44100 * 5)
        self.time_data = []
        self.spectrogram_data = []

        # Initialize sample counter
        self.sample_counter = 0

        self.running = True  # Set running flag to True

        try:
            # Start the audio input stream
            with sd.InputStream(
                    channels=1,
                    callback=self.audio_callback,
                    samplerate=self.sample_rate,
                    dtype='float32',
                    blocksize=int(self.sample_rate * 0.05)  # 50ms chunks for smoother updates
            ):
                while self.running:
                    # Update visualizations
                    if self.show_waveform:
                        self.update_waveform_plot()
                    if self.show_fft:
                        self.update_fft_plot()
                    if self.show_spectrogram:
                        self.update_spectrogram()

                    # Redraw the figure canvas
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                    # Sleep briefly to yield control
                    time.sleep(0.000001)
        except Exception as e:
            # Handle any exceptions during streaming
            print(f"An error occurred during streaming: {e}")
            self.running = False

    def main(self):
        """
        Main function to start the audio visualizer.
        """
        try:
            # Prompt user for audio source
            print("Choose audio source:")
            print("1. Live audio from microphone")
            print("2. SRT audio stream via FFmpeg")
            source_choice = input("Enter '1' for live audio or '2' for SRT stream: ").strip()

            # Prompt for SRT details if selected
            if source_choice == '2':
                input_srt = input("Enter the SRT stream URL (e.g., 'srt://192.168.1.157:5000?mode=caller'): ").strip()
                output_file = input("Enter the file path to save the recording (e.g., 'recorded_audio.wav'): ").strip()
                print("Starting SRT streaming and saving...")
                stream_and_save_srt(input_srt, output_file)
                return  # Exit after streaming

            # Confirm and set audio source
            if source_choice == '1':
                print("Selected live audio stream from microphone.")
            else:
                print("Invalid choice for audio source. Exiting.")
                return

            # Prompt user for visualizations to display
            print("Choose visualizations to display:")
            self.show_waveform = input("Show waveform? (y/n): ").strip().lower() == 'y'
            self.show_fft = input("Show FFT? (y/n): ").strip().lower() == 'y'
            self.show_spectrogram = input("Show spectrogram? (y/n): ").strip().lower() == 'y'

            # Ensure at least one visualization is selected
            if not any([self.show_waveform, self.show_fft, self.show_spectrogram]):
                print("No visualizations selected. Exiting.")
                return

            # Prompt user for frequency range
            print("Please select the frequency range for visualization (Spectrogram and FFT):")
            print("1. 0-20,000 Hz (default)")
            print("2. 0-5,000 Hz")
            print("3. 0-2,000 Hz")
            print("4. 0-1,000 Hz")
            print("5. 0-500 Hz")
            freq_choice = input("Enter the number of your choice: ").strip()

            freq_options = {
                '1': (0, 20000),
                '2': (0, 5000),
                '3': (0, 2000),
                '4': (0, 1000),
                '5': (0, 500)
            }
            if freq_choice in freq_options:
                self.MIN_FREQ, self.MAX_FREQ = freq_options[freq_choice]
            else:
                print("Invalid choice for frequency range. Using default 0-20,000 Hz.")
                self.MIN_FREQ, self.MAX_FREQ = 0, 20000
            print(f"Frequency range set to {self.MIN_FREQ}-{self.MAX_FREQ} Hz.")

            # Prompt user for time window
            print("Select the time window (in seconds) for visualization:")
            print("1. 3 seconds")
            print("2. 5 seconds")
            print("3. 7 seconds")
            print("4. 9 seconds")
            time_window_choice = input("Enter your choice (1-4): ").strip()

            time_window_options = {
                '1': 3,
                '2': 5,
                '3': 7,
                '4': 9
            }
            if time_window_choice in time_window_options:
                self.TIME_WINDOW = time_window_options[time_window_choice]
            else:
                print("Invalid choice. Using default 5 seconds.")
                self.TIME_WINDOW = 5
            print(f"Time window set to {self.TIME_WINDOW} seconds.")

            # Set up the plots based on selected visualizations
            self.setup_plots()

            # Start the appropriate audio stream
            print("Starting live audio stream...")
            self.start_live_audio_stream()

        except KeyboardInterrupt:
            # Handle user interruption
            print("Interrupted by user.")
            self.running = False
        except Exception as e:
            # Handle any other exceptions
            print("An error occurred:", e)
            self.running = False

# Run the AudioVisualizer if this script is executed
if __name__ == "__main__":
    # Create an instance of the AudioVisualizer class
    visualizer = AudioVisualizer()
    # Run the main function
    visualizer.main()