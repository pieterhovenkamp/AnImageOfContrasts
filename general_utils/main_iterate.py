#!/usr/bin/env python3

import datetime
import time


def main_iterate(function, num, temp_file_path, temp_file_name='temp_sleep_output.txt', sleep_time=3600,):
    """
    We use this function to make a queue of scripts that execute the function <function>.

    The function should be such that it stops itself after several iterations, then in the current function output is written
    to a txt-file that this iteration of function() is finished. By simultaneously running another script with
    num=<next>), it is checked every sleep_time whether the previous run is finished, and if so a new run is started.

    tl;dr:
    We build a queue of our <function> where the next iteration only starts when the previous one is finished. We do
    this by making scripts main1.py with main_iterate(function, 1), main2.py with main_iterate(function, 2) etc. Start these all
    simultaneously and everything will be arranged.

    """
    num = int(num)
    path = temp_file_path / temp_file_name

    if num == 1:
        # We create a new empty output file at the start of the first iteration
        with open(path, "w") as file:
            pass

    elif num > 1:
        print(f"main({num}) - we sleep until the output of main({num - 1}) is there")
        while True:
            print(f"main({num})", datetime.datetime.now(), " - Checking for output..")
            with open(path, "r") as file:
                lines = [line.strip() for line in file]

            if f"finished_run{num - 1}" in lines:
                print(f"main({num})", datetime.datetime.now(), " - We should start!")
                break
            print(f"main({num})", datetime.datetime.now(), " - No output yet")

            time.sleep(sleep_time)

    print(f"Starting {function.__name__} iteration {num}")

    try:
        function()
    except Exception as e:
        print(f"iteration {num} gave the following error: {e}")
    finally:
        # If the program is finished, we write a message to an output file
        with open(path, "a") as file:
            print(f"Finished {function.__name__} iteration {num}!")
            file.write(f"finished_run{num} \n")


if __name__ == "__main__":
    def function_dummy():
        print("Dummy function")


    main_iterate(function_dummy, 1, temp_file_path='/Users/pieter/Documents/databases')