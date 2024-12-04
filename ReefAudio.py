#!/usr/bin/env python3

"""
This script allows the user to specify actions such as:
- Record
- Listen
- Listen & Visualize
- Listen & Record & Visualize

The script executes the specified action.
"""

import sys
import subprocess
import datetime

def is_int(user_input):
    try:
        int(user_input)
        return True
    except ValueError:
        print("\nPlease enter a valid integer!\n")
        return False

def is_in_range(user_input, choices):
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

    if action_choice == 1:
        # Record
        print("Starting recording...")
        target_address = input("Enter the target SRT address (e.g., 'srt://<IP_ADDRESS>:8000?mode=caller'): ").strip()
        duration = input("Enter recording duration in seconds: ").strip()
        if not is_int(duration):
            print("Invalid duration. Exiting.")
            sys.exit(1)
        duration = int(duration)
        output_file = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        # Build the ffmpeg command
        ffmpeg_command = f'ffmpeg -i "{target_address}" -t {duration} -c:a pcm_s16le "{output_file}"'

        try:
            subprocess.run(ffmpeg_command, shell=True)
            print(f"Recording saved to {output_file}")
        except Exception as e:
            print(f"Error during recording: {e}")

    elif action_choice == 2:
        # Listen
        print("Starting to listen...")
        target_address = input("Enter the target SRT address (e.g., 'srt://<IP_ADDRESS>:8000?mode=caller'): ").strip()
        try:
            subprocess.run(["ffplay", target_address])
        except Exception as e:
            print(f"Error during listening: {e}")

    elif action_choice == 3:
        # Listen & Visualize
        print("Starting to listen and visualize...")
        target_address = input("Enter the target SRT address (e.g., 'srt://<IP_ADDRESS>:8000?mode=caller'): ").strip()

        # Use the exact ffmpeg command as specified
        ffmpeg_command = f'ffmpeg -i "{target_address}" -c:a pcm_s16le -f tee -map 0:a ' \
                         f'"[f=wav]output.wav|[f=s16le]pipe:1"'
        ffplay_command = 'ffplay -f s16le -ar 44100 -i pipe:0'

        # Start the pipeline
        try:
            ffmpeg_process = subprocess.Popen(ffmpeg_command, shell=True, stdout=subprocess.PIPE)
            ffplay_process = subprocess.Popen(ffplay_command, shell=True, stdin=ffmpeg_process.stdout)
            ffmpeg_process.stdout.close()
            ffplay_process.communicate()
        except Exception as e:
            print(f"Error during listening and visualization: {e}")

    elif action_choice == 4:
        # Listen & Record & Visualize
        print("Starting to listen, record, and visualize...")
        target_address = input("Enter the target SRT address (e.g., 'srt://<IP_ADDRESS>:8000?mode=caller'): ").strip()

        # Use the exact ffmpeg command as specified
        output_file = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        ffmpeg_command = f'ffmpeg -i "{target_address}" -c:a pcm_s16le -f tee -map 0:a ' \
                         f'"[f=wav]{output_file}|[f=s16le]pipe:1"'
        ffplay_command = 'ffplay -f s16le -ar 44100 -i pipe:0'

        # Start the pipeline
        try:
            ffmpeg_process = subprocess.Popen(ffmpeg_command, shell=True, stdout=subprocess.PIPE)
            ffplay_process = subprocess.Popen(ffplay_command, shell=True, stdin=ffmpeg_process.stdout)
            ffmpeg_process.stdout.close()
            ffplay_process.communicate()
        except Exception as e:
            print(f"Error during listening, recording, and visualization: {e}")

if __name__ == "__main__":
    main()
