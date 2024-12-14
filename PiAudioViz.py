#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from collections import deque
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import subprocess
import threading
import queue
import pyaudio
import wave
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

    def stream_and_save_srt(self, input_srt, output_file, save_audio=True, visualize=True, playback=True):
        """
        Stream SRT audio, save to a file, play through speakers, and visualize audio.

        Parameters:
        - input_srt (str): The SRT stream URL (e.g., 'srt://128.171.120.197:8000?mode=caller').
        - output_file (str): Path to save the audio file.
        - save_audio (bool): Whether to save the audio to a file.
        - visualize (bool): Whether to visualize the audio stream.
        - playback (bool): Whether to play the audio through speakers.
        """

        # Parse and update the SRT URL with required parameters
        url_parts = urlparse(input_srt)
        query_params = parse_qs(url_parts.query)
        required_params = {'latency': '5000', 'pkt_size': '1316', 'buffer_size': '100000'}
        for param, default_value in required_params.items():
            if param not in query_params:
                query_params[param] = [default_value]
        input_srt_with_params = urlunparse(url_parts._replace(query=urlencode(query_params, doseq=True)))

        # FFmpeg command for streaming
        ffmpeg_command = f'ffmpeg -re -i "{input_srt_with_params}" -c:a pcm_s16le -f s16le -buffer_size 131072 -'

        try:
            # Start the FFmpeg process to read raw PCM audio
            process = subprocess.Popen(
                ffmpeg_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 8
            )

            # Open file for saving the raw audio
            if save_audio:
                wav_file = wave.open(output_file, 'wb')
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
                wav_file.setframerate(44100)

            # Setup PyAudio for playback
            if playback:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,  # Set this to the SRT stream's sample rate
                    output=True
                )

            def read_ffmpeg_stderr(pipe):
                """Log FFmpeg stderr."""
                while True:
                    line = pipe.readline()
                    if not line:
                        break
                    print(f"FFmpeg stderr: {line.decode('utf-8')}", end='')

            # Start a thread to monitor FFmpeg's stderr
            stderr_thread = threading.Thread(target=read_ffmpeg_stderr, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()

            while self.running:
                # Read a chunk of audio data from the FFmpeg process
                raw_audio = process.stdout.read(self.FFT_WINDOW_SIZE * 2)  # 16-bit PCM, 2 bytes per sample
                if not raw_audio:
                    print("No audio data received. Exiting.")
                    break

                # Save audio
                if save_audio:
                    wav_file.writeframes(raw_audio)

                # Playback audio
                if playback:
                    stream.write(raw_audio)

                # Process audio for visualization
                if visualize:
                    try:
                        # Convert raw byte data to a NumPy array for visualization
                        audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / np.iinfo(
                            np.int16).max
                        # Pass it to the audio callback
                        self.audio_callback(audio_data.reshape(-1, 1), len(audio_data), None, None)
                        if self.show_waveform:
                            self.update_waveform_plot()
                        if self.show_fft:
                            self.update_fft_plot()
                        if self.show_spectrogram:
                            self.update_spectrogram()

                        self.fig.canvas.draw_idle()
                        self.fig.canvas.flush_events()
                    except Exception as e:
                        print(f"Error processing visualization: {e}")

            process.stdout.close()
            process.stderr.close()
            process.wait()

            if playback:
                stream.stop_stream()
                stream.close()
                p.terminate()

            if save_audio:
                wav_file.close()

        except KeyboardInterrupt:
            print("Interrupted by user. Stopping...")
            process.terminate()
            if save_audio:
                wav_file.close()
            if playback:
                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            print(f"Error during SRT stream: {e}")
            process.terminate()
            if save_audio:
                wav_file.close()
            if playback:
                stream.stop_stream()
                stream.close()
                p.terminate()


    def setup_plots(self):
        """
        Set up the Matplotlib plots for the selected visualizations.
        """
        # Enable interactive plotting mode
        plt.ion()

        # Determine the number of subplots needed based on user selection
        num_plots = sum([self.show_waveform, self.show_fft, self.show_spectrogram])
        # Create a figure and the required number of axes (subplots)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        # Ensure axes is iterable even if there's only one plot
        if num_plots == 1:
            axes = [axes]

        self.fig = fig  # Reference to the figure object
        plot_index = 0  # Index for accessing axes

        # Set up waveform plot if selected
        if self.show_waveform:
            self.ax_waveform = axes[plot_index]
            # Initialize an empty line plot for waveform
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
            # Initialize an empty line plot for FFT
            self.line_fft, = self.ax_fft.plot([], [], lw=1.5)
            # Set plot labels and title
            self.ax_fft.set_xlabel('Frequency (Hz)')
            self.ax_fft.set_ylabel('Amplitude')
            self.ax_fft.set_title("Real-Time FFT Visualization")
            # Set x-axis limits to frequency range
            self.ax_fft.set_xlim(self.MIN_FREQ, self.MAX_FREQ)
            # Set y-axis limits to amplitude range
            self.ax_fft.set_ylim(self.MIN_AMP, self.MAX_AMP)
            plot_index += 1  # Move to next subplot

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
            plot_index += 1  # Move to next subplot

        # Adjust the layout to prevent overlap of subplots
        self.fig.tight_layout()
        # Draw the initial plots to display empty visualizations
        self.fig.canvas.draw_idle()

    def update_waveform_plot(self):
        """
        Update the waveform plot with the latest audio data.
        """
        try:
            # Get all available audio data from the queue
            while not self.audio_queue.empty():
                audio_data, timestamps = self.audio_queue.get_nowait()
                # Extend the deque with new audio data
                self.waveform_data.extend(audio_data)
                # Extend the deque with new timestamps
                self.waveform_time.extend(timestamps)
        except Exception as e:
            print(f"Error updating waveform plot: {e}")

        if self.waveform_data and self.waveform_time:
            # Use data within the sliding time window
            current_time = self.waveform_time[-1]
            min_time = current_time - self.TIME_WINDOW
            # Find indices within the time window
            indices = np.where((np.array(self.waveform_time) >= min_time))[0]
            data = np.array(self.waveform_data)[indices]
            time_axis = np.array(self.waveform_time)[indices]

            # Update the waveform plot with the data within the time window
            self.line_waveform.set_data(time_axis, data)
            self.ax_waveform.set_xlim(min_time, current_time)
            self.ax_waveform.set_ylim(-1.0, 1.0)  # Ensure amplitude range is consistent

    def update_fft_plot(self):
        """
        Update the FFT plot with the latest frequency data.
        """
        try:
            # Check if frequencies have been initialized
            if self.frequencies is None:
                return  # Frequencies not yet initialized; skip update

            # Get the latest FFT magnitude data from the queue
            y_magnitude = None
            while not self.fft_queue.empty():
                y_magnitude = self.fft_queue.get_nowait()

            if y_magnitude is None:
                return  # No FFT data available; skip update

            # Limit frequencies to the selected range
            freq_indices = np.where((self.frequencies >= self.MIN_FREQ) & (self.frequencies <= self.MAX_FREQ))
            frequencies = self.frequencies[freq_indices]
            y_magnitude = y_magnitude[freq_indices]

            # Normalize amplitude to be between MIN_AMP and MAX_AMP
            max_magnitude = np.max(y_magnitude)
            if max_magnitude == 0:
                max_magnitude = 1e-6  # Avoid division by zero
            y_magnitude_normalized = y_magnitude / max_magnitude * self.MAX_AMP

            # Update the FFT plot with the normalized data
            self.line_fft.set_data(frequencies, y_magnitude_normalized)
            self.ax_fft.set_xlim(self.MIN_FREQ, self.MAX_FREQ)
            self.ax_fft.set_ylim(self.MIN_AMP, self.MAX_AMP)
        except Exception as e:
            print(f"Error updating FFT plot: {e}")

    def update_spectrogram(self):
        """
        Update the spectrogram with the latest FFT data.
        """
        try:
            # Check if frequencies have been initialized
            if self.frequencies is None:
                return  # Frequencies not yet initialized; skip update

            # Get the latest FFT data for spectrogram
            while not self.spectrogram_queue.empty():
                fft_data, current_time = self.spectrogram_queue.get_nowait()
                # Append the new FFT data and timestamp
                self.spectrogram_data.append(fft_data)
                self.time_data.append(current_time)
        except Exception as e:
            print(f"Error updating spectrogram: {e}")

        if len(self.spectrogram_data) > 1 and len(self.time_data) > 1:
            # Convert spectrogram data to numpy array and transpose
            spec_data = np.array(self.spectrogram_data).T

            # Convert amplitude to decibel scale for better visualization
            spec_data_db = 10 * np.log10(spec_data + 1e-6)

            # Clip data for better visualization contrast
            spec_data_db = np.clip(spec_data_db, a_min=-20, a_max=0)

            # Limit frequencies to the selected range
            freq_bins = self.frequencies
            freq_indices = np.where((freq_bins >= self.MIN_FREQ) & (freq_bins <= self.MAX_FREQ))
            freq_bins = freq_bins[freq_indices]
            spec_data_db = spec_data_db[freq_indices]

            # Limit time data to a sliding window based on TIME_WINDOW
            current_time = self.time_data[-1]
            min_time = current_time - self.TIME_WINDOW  # Sliding time window
            time_indices = np.where((np.array(self.time_data) >= min_time))[0]
            time_axis = np.array(self.time_data)[time_indices]
            spec_data_db = spec_data_db[:, time_indices]

            # Update the spectrogram image with the new data
            self.img_spectrogram.set_data(spec_data_db)
            self.img_spectrogram.set_extent([
                min_time,        # Start time of the spectrogram
                current_time,    # End time of the spectrogram
                self.MIN_FREQ,   # Minimum frequency
                self.MAX_FREQ    # Maximum frequency
            ])
            self.img_spectrogram.set_cmap('inferno')  # Set colormap to 'inferno' for high contrast
            self.ax_spectrogram.set_xlim(min_time, current_time)
            self.ax_spectrogram.set_ylim(self.MIN_FREQ, self.MAX_FREQ)

            # Redraw the canvas to reflect updates
            self.fig.canvas.draw_idle()

    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function for processing incoming audio data.

        Parameters:
        - indata: Input audio data array.
        - frames: Number of frames in the buffer.
        - time_info: Dictionary containing timing information.
        - status: Status of the stream.
        """
        try:
            if status:
                # Print any stream status messages
                print("Stream status:", status)

            # Make a copy of the audio data from the first channel
            audio_data = indata[:, 0].copy()

            # Save current sample counter for timestamp calculations
            current_sample_counter = self.sample_counter

            # Generate continuous timestamps for the audio data
            timestamps = (np.arange(len(audio_data)) + current_sample_counter) / self.sample_rate

            # Increment sample counter by the number of frames processed
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

                # Apply a window function (Hanning window) to reduce spectral leakage
                window = np.hanning(len(audio_data_padded))
                audio_windowed = audio_data_padded * window

                # Compute the one-sided FFT (positive frequencies only)
                yf = np.abs(np.fft.rfft(audio_windowed))
                # Compute frequencies if not already done
                if self.frequencies is None:
                    self.frequencies = np.fft.rfftfreq(self.FFT_WINDOW_SIZE, d=1 / self.sample_rate)

                # Current time for spectrogram and FFT plots
                fft_time = fft_sample_counter / self.sample_rate

                # Put the FFT magnitude data into the FFT queue
                if self.show_fft:
                    self.fft_queue.put(yf)

                # Put the spectrogram data into the spectrogram queue
                if self.show_spectrogram:
                    self.spectrogram_queue.put((yf, fft_time))
        except Exception as e:
            print(f"Error in audio_callback: {e}")

    def start_live_audio_stream(self):
        """
        Start streaming live audio from the microphone and update visualizations.
        """
        # Dynamically determine the sample rate from the default input device
        if self.sample_rate is None:
            try:
                default_input_device = sd.default.device[0]
                if default_input_device is None or default_input_device < 0:
                    default_input_device = sd.query_devices(kind='input')['index']
                input_device_info = sd.query_devices(default_input_device, 'input')
                self.sample_rate = int(input_device_info['default_samplerate'])
                print(f"Detected sample rate: {self.sample_rate} Hz")
            except Exception as e:
                print(f"Error detecting default input device: {e}")
                return

        # Adjust FFT_WINDOW_SIZE for desired resolution (e.g., 1/10th of a second)
        self.FFT_WINDOW_SIZE = int(self.sample_rate * 0.1)  # For a 100ms window
        # Ensure FFT_WINDOW_SIZE is a power of two
        self.FFT_WINDOW_SIZE = 2 ** int(np.log2(self.FFT_WINDOW_SIZE))
        print(f"FFT window size: {self.FFT_WINDOW_SIZE} samples")

        # Initialize buffers for waveform and spectrogram data
        maxlen = int(self.TIME_WINDOW * self.sample_rate)
        self.waveform_data = deque(maxlen=maxlen)
        self.waveform_time = deque(maxlen=maxlen)
        self.time_data = []           # Timestamps for spectrogram
        self.spectrogram_data = []    # FFT data for spectrogram

        # Initialize sample counter
        self.sample_counter = 0

        self.running = True  # Set running flag to True to start the loop

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
                    # Update visualizations as per user selection
                    if self.show_waveform:
                        self.update_waveform_plot()
                    if self.show_fft:
                        self.update_fft_plot()
                    if self.show_spectrogram:
                        self.update_spectrogram()

                    # Redraw the figure canvas to reflect updates
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                    # Sleep briefly to yield control and allow GUI to update
                    plt.pause(0.001)  # Use plt.pause to allow GUI event loop to process
        except Exception as e:
            # Handle any exceptions during streaming
            print(f"An error occurred during streaming: {e}")
            self.running = False  # Stop the loop in case of error

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

            # Confirm and set audio source
            if source_choice == '1':
                print("Selected live audio stream from microphone.")
            elif source_choice == '2':
                print("Selected SRT audio stream via FFmpeg.")
                # Prompt for SRT details if selected
                input_srt = input("Enter the SRT stream URL (e.g., 'srt://128.171.120.197:8000?mode=caller'): ").strip()
                output_file = input("Enter the file path to save the recording (e.g., 'recorded_audio.wav'): ").strip()
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

            if source_choice == '1':
                # Start the live audio stream from microphone
                print("Starting live audio stream...")
                self.start_live_audio_stream()
            elif source_choice == '2':
                # Dynamically determine the sample rate (needed before streaming)
                if self.sample_rate is None:
                    try:
                        self.sample_rate = 44100  # Set a default sample rate or retrieve from stream
                        print(f"Using sample rate: {self.sample_rate} Hz")
                    except Exception as e:
                        print(f"Error setting sample rate: {e}")
                        return

                # Adjust FFT_WINDOW_SIZE for desired resolution
                self.FFT_WINDOW_SIZE = int(self.sample_rate * 0.1)  # For a 100ms window
                # Ensure FFT_WINDOW_SIZE is a power of two
                self.FFT_WINDOW_SIZE = 2 ** int(np.log2(self.FFT_WINDOW_SIZE))
                print(f"FFT window size: {self.FFT_WINDOW_SIZE} samples")

                # Initialize buffers for waveform and spectrogram data
                maxlen = int(self.TIME_WINDOW * self.sample_rate)
                self.waveform_data = deque(maxlen=maxlen)
                self.waveform_time = deque(maxlen=maxlen)
                self.time_data = []           # Timestamps for spectrogram
                self.spectrogram_data = []    # FFT data for spectrogram

                # Initialize sample counter
                self.sample_counter = 0

                # Start streaming and visualizing SRT audio
                print("Starting SRT streaming and visualization...")
                self.running = True
                self.stream_and_save_srt(input_srt, output_file)
        except KeyboardInterrupt:
            # Handle user interruption gracefully
            print("Interrupted by user.")
            self.running = False
        except Exception as e:
            # Handle any other exceptions
            print("An error occurred:", e)
            self.running = False

# Run the AudioVisualizer if this script is executed directly
if __name__ == "__main__":
    # Create an instance of the AudioVisualizer class
    visualizer = AudioVisualizer()
    # Run the main function to start the visualizer
    visualizer.main()