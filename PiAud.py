#!/usr/bin/env python3

"""
pyaud.py

This script provides functions to list input audio devices, record audio, and play back audio.
"""

import pyaudio
import wave
from datetime import datetime, timedelta
import time
import os
import json

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    input_devices = 0
    devices = {}

    for i in range(0, numdevices):
        # Check if device has an input channel
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            input_devices += 1
            device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            devices[i] = device_name

    p.terminate()
    return input_devices, devices

def record_audio(device_index=1, duration=10, sample_rate=96000, output_directory=".", prefix="output"):
    p = pyaudio.PyAudio()

    # Try-finally block used so that stream gets closed if error occurs
    try:
        stream = p.open(format=pyaudio.paInt16,  # Set to 16-bit audio format
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024)

        frames = []

        print(f"Recording audio with sample rate {sample_rate}, duration {duration}s")

        # Calculate total frames to read
        total_frames = int((sample_rate / 1024) * duration)

        for i in range(0, total_frames):
            # Reads 1024 frames of audio data from input audio stream
            data = stream.read(1024, exception_on_overflow=False)
            # Appends chunk to audio data
            frames.append(data)

        print("Recording complete.")

        # Generate a file name based on the current date and time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"{prefix}_{timestamp}.wav"
        file_path = os.path.join(output_directory, file_name)

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # set to mono audio
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))  # concatenates audio data stored in frames list into single byte string

        print(f"Recording saved as: {file_path}")

    except KeyboardInterrupt:
        print("Recording stopped by keyboard interrupt")
        pass

    finally:
        # Close stream and remove pyaudio object
        stream.stop_stream()
        stream.close()
        p.terminate()

def play_audio(file_path):
    p = pyaudio.PyAudio()

    try:
        wf = wave.open(file_path, 'rb')

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)

        print("Playing audio...")

        while data:
            stream.write(data)
            data = wf.readframes(1024)

        print("Playback complete.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
