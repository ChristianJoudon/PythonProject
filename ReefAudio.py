#!/usr/bin/env python3

"""
This script allows the user to:
- Record
- Listen
- Listen & Visualize
- Listen & Record & Visualize

It integrates with the AudioVisualizer class for real-time visualization.
"""

import sys
import subprocess
import datetime
from AudioVisualizer import AudioVisualizer  # Assuming AudioVisualizer is in the same directory

def is_int(user_input):
    """Check if input is an integer."""
    try:
        int(user_input)
        return True
    except ValueError:
        print("\nPlease enter a valid integer!\n")
        return False

def is_in_range(user_input, choices):
    """Check if input is within the valid range."""
    if 0 < user_input <= choices:
        return True
    else:
        print(f"\nPlease enter an integer between 1 and {choices}\n")
        return False

def main():
    # Present the user with action options
    print("Choose an action:")
    print("   [1] - Record")
    print("   [2] - Listen")
    print("   [3] - Listen & Visualize")
    print("   [4] - Listen & Record & Visualize")
    action_choice = input()

    if not is_int(action_choice) or not is_in_range(int(action_choice), 4):
        print("Invalid choice. Exiting.")
        sys.exit(1)

    action_choice = int(action_choice)

    # Common inputs
    target_address = input("Enter the target SRT address (e.g., 'srt://<IP_ADDRESS>:8000?mode=caller'): ").strip()
    output_file = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    if action_choice == 1:
        # Record only
        print("Starting recording...")
        duration = input("Enter recording duration in seconds: ").strip()
        if not is_int(duration):
            print("Invalid duration. Exiting.")
            sys.exit(1)
        duration = int(duration)

        ffmpeg_command = f'ffmpeg -i "{target_address}" -t {duration} -c:a pcm_s16le "{output_file}"'

        try:
            subprocess.run(ffmpeg_command, shell=True)
            print(f"Recording saved to {output_file}")
        except Exception as e:
            print(f"Error during recording: {e}")

    elif action_choice == 2:
        # Listen only
        print("Starting to listen...")
        try:
            subprocess.run(["ffplay", target_address])
        except Exception as e:
            print(f"Error during listening: {e}")

    elif action_choice == 3:
        # Listen & Visualize
        print("Starting to listen and visualize...")
        visualizer = AudioVisualizer()
        visualizer.show_waveform = True
        visualizer.show_fft = True
        visualizer.show_spectrogram = True

        # Configure visualizer and start streaming
        try:
            visualizer.setup_plots()
            visualizer.stream_and_save_srt(input_srt=target_address, output_file=None, save_audio=False, visualize=True, playback=True)
        except Exception as e:
            print(f"Error during listening and visualization: {e}")

    elif action_choice == 4:
        # Listen & Record & Visualize
        print("Starting to listen, record, and visualize...")
        visualizer = AudioVisualizer()
        visualizer.show_waveform = True
        visualizer.show_fft = True
        visualizer.show_spectrogram = True

        # Configure visualizer and start streaming
        try:
            visualizer.setup_plots()
            visualizer.stream_and_save_srt(input_srt=target_address, output_file=output_file, save_audio=True, visualize=True, playback=True)
            print(f"Recording saved to {output_file}")
        except Exception as e:
            print(f"Error during listening, recording, and visualization: {e}")

if __name__ == "__main__":
    main()
