#!/usr/bin/env python3

"""
PiAudioConfig.py

This script runs on the Raspberry Pi to configure audio recording settings.
It allows the user to set the sample rate, recording duration, number of times per hour,
and starts the recording accordingly.
"""

import sys
import json
import datetime
from datetime import datetime, timedelta
import time
import os
from pyaud import list_audio_devices, record_audio

# Input Validation Functions
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

def valid_datetime():
    while True:
        print("Enter datetime in the following format: YYYY-MM-DD HH:MM:SS")
        user_input = input()

        try:
            # Parse the user input into a datetime object
            user_datetime = datetime.strptime(user_input, '%Y-%m-%d %H:%M:%S')
            return user_datetime
        except ValueError:
            print("\nPlease enter a valid datetime!\n")

def valid_time():
    while True:
        print("Enter time in the following format: HH:MM:SS")
        user_input = input()

        try:
            # Parse the user input into a time object
            time_obj = datetime.strptime(user_input, '%H:%M:%S').time()
            return time_obj
        except ValueError:
            print("\nPlease enter a valid time!\n")

def time_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def set_sample_rate():
    while True:
        num_choices = 4

        print("Choose sample rate:")
        print("   [1] - 44.1 kHz")
        print("   [2] - 48 kHz")
        print("   [3] - 88.2 kHz")
        print("   [4] - 96 kHz")
        sample_rate = input()

        if not is_int(sample_rate):
            continue

        if not is_in_range(int(sample_rate), num_choices):
            continue
        else:
            sample_rate = int(sample_rate)
            if sample_rate == 1:
                return 44100
            elif sample_rate == 2:
                return 48000
            elif sample_rate == 3:
                return 88200
            elif sample_rate == 4:
                return 96000

def set_duration():
    while True:
        num_choices = 4

        print("Choose a recording duration:")
        print("   [1] - 1 minute")
        print("   [2] - 2 minutes")
        print("   [3] - 5 minutes")
        print("   [4] - 10 minutes")
        duration_choice = input()

        if not is_int(duration_choice):
            continue

        if not is_in_range(int(duration_choice), num_choices):
            continue
        else:
            duration_choice = int(duration_choice)

            if duration_choice == 1:
                duration = 60  # seconds
            elif duration_choice == 2:
                duration = 120
            elif duration_choice == 3:
                duration = 300
            elif duration_choice == 4:
                duration = 600
            break

    return duration

def set_times_per_hour():
    while True:
        num_choices = 4

        print("Specify how many times per hour to record:")
        print("   [1] - 1 time per hour")
        print("   [2] - 2 times per hour")
        print("   [3] - 3 times per hour")
        print("   [4] - 4 times per hour")
        times_choice = input()

        if not is_int(times_choice):
            continue

        if not is_in_range(int(times_choice), num_choices):
            continue
        else:
            times_choice = int(times_choice)
            return times_choice  # returns 1,2,3,4

def set_period_between_recordings(duration, times_per_hour):
    while True:
        print("Do you want to specify the period between recordings? (y/n)")
        user_input = input().strip().lower()

        if user_input == 'y':
            # User wants to specify period between recordings
            while True:
                period_time = valid_time()
                period_seconds = time_to_seconds(period_time)
                if period_seconds <= 0:
                    print("Period must be greater than zero.")
                    continue
                return period_seconds
        elif user_input == 'n':
            # Calculate period so that recordings are equidistant within the hour
            total_recording_time = duration * times_per_hour
            if total_recording_time > 3600:
                print("\nTotal recording time exceeds one hour. Please adjust duration or times per hour.\n")
                return None
            remaining_time = 3600 - total_recording_time
            period_seconds = remaining_time / times_per_hour
            return period_seconds
        else:
            print("\nInvalid input. Please enter 'y' or 'n'.\n")

def set_start_time():
    while True:
        num_choices = 3

        print("Choose start time:")
        print("   [1] - Now")
        print("   [2] - 1 hour from now")
        print("   [3] - Specify date and time")
        start_time = input()

        if not is_int(start_time):
            continue

        if not is_in_range(int(start_time), num_choices):
            continue
        else:
            start_time = int(start_time)

            # Grabs current time for efficient time delta calculation
            current_time = datetime.now()

            if start_time == 1:
                return current_time
            elif start_time == 2:
                one_hour_later = current_time + timedelta(hours=1)
                return one_hour_later
            else:
                return valid_datetime()

def set_end_time(start_time):
    while True:
        num_choices = 3

        print("Choose end time:")
        print("   [1] - 1 hour after first recording")
        print("   [2] - 2 hours after first recording")
        print("   [3] - Specify date and time")
        user_choice = input()

        if not is_int(user_choice):
            continue

        if not is_in_range(int(user_choice), num_choices):
            continue
        else:
            user_choice = int(user_choice)

            if user_choice == 1:
                end_time = start_time + timedelta(hours=1)
            elif user_choice == 2:
                end_time = start_time + timedelta(hours=2)
            else:
                end_time = valid_datetime()

            if end_time <= start_time:
                print("\nEnd time must be after start time.\n")
                continue

            return end_time

def set_recording_device():
    print("Choose a recording device:\n")
    num_choices, devices = list_audio_devices()

    if num_choices == 0:
        print("No input devices found, terminating program.")
        sys.exit(1)

    for idx, name in devices.items():
        print(f"   [{idx}] - {name}")

    while True:
        user_choice = input()

        if not is_int(user_choice):
            continue

        if not is_in_range(int(user_choice), num_choices):
            continue

        else:
            device = int(user_choice)
            break

    return device

def main():
    # Set sample rate
    sample_rate = set_sample_rate()

    # Set recording duration
    duration = set_duration()  # in seconds

    # Set times per hour
    times_per_hour = set_times_per_hour()

    # Set period between recordings
    period_seconds = set_period_between_recordings(duration, times_per_hour)
    if period_seconds is None:
        print("Cannot schedule recordings with the given parameters. Exiting.")
        sys.exit(1)

    # Set start and end times
    start_time = set_start_time()
    end_time = set_end_time(start_time)

    # Set recording device
    device_index = set_recording_device()

    # Save configuration to JSON
    config = {
        "sample_rate": sample_rate,
        "duration": duration,
        "times_per_hour": times_per_hour,
        "period_between_recordings": period_seconds,
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "device_index": device_index,
        "output_directory": os.getcwd(),
        "prefix": "recording"
    }

    with open("config.json", "w") as outfile:
        json.dump(config, outfile, default=str)

    print("\nConfiguration saved to config.json")
    print("\nStarting scheduled recordings...")

    # Start scheduled recordings
    current_time = datetime.now()
    if start_time > current_time:
        wait_seconds = (start_time - current_time).total_seconds()
        print(f"Waiting until start time ({start_time})...")
        time.sleep(wait_seconds)

    # Calculate the times at which recordings should start
    recording_times = []
    next_recording_time = start_time

    while next_recording_time < end_time:
        for i in range(times_per_hour):
            if next_recording_time >= end_time:
                break
            recording_times.append(next_recording_time)
            next_recording_time += timedelta(seconds=duration + period_seconds)
        # Adjust to the start of the next hour
        next_hour = (next_recording_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        next_recording_time = next_hour

    # Start scheduled recordings
    for recording_time in recording_times:
        current_time = datetime.now()
        if recording_time > current_time:
            wait_seconds = (recording_time - current_time).total_seconds()
            print(f"\nWaiting until next recording at {recording_time}...")
            time.sleep(wait_seconds)

        # Start recording
        print(f"\nStarting recording at {recording_time} for {duration} seconds...")

        record_audio(
            device_index=device_index,
            duration=duration,
            sample_rate=sample_rate,
            output_directory=config["output_directory"],
            prefix=config["prefix"]
        )

    print("\nAll scheduled recordings completed.")

if __name__ == "__main__":
    main()
