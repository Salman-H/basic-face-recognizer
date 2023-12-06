#!/usr/bin/env python

"""
A program for detecting and recognizing faces in video streams

USAGE:
    python main.py <name> <max_conf> <video_source>

AUTHOR:
    Salman Hashmi
    sah517@g.harvard.edu
"""
import sys
import os
import re
from recognizer import train, process_video


def main():
    # handle user input
    if len(sys.argv) != 4:
        print("\nUsage: python main.py <name> <max_conf> <video_source>")
        print("example: python main.py 'obama' 50.0 'obama-speech-4.mp4")
        exit(1)

    # check if the specified person's training image set exists
    names = [name for name in os.listdir('./training-set') if os.path.isdir(os.path.join('./training-set', name))]
    if sys.argv[1] not in names:
        print("\n{}'s training set does not exit in the training-set directory.".format(sys.argv[1]))
        exit(2)

    # check if the max confidence value is a float between 0.0 and 100.0
    if re.match("^\d+?\.\d+?$", sys.argv[2]) is None or 100.0 < sys.argv[2] < 0.0:
        print("\nmax_conf must be a float between 0.0 and 100.0")
        exit(3)

    # check if the video file exists
    videos = [video for video in os.listdir('./videos-input') if os.path.isfile(os.path.join('./videos-input', video))]
    if sys.argv[3] not in videos:
        print("\n{} does not exist in the videos-input directory.".format(sys.argv[3]))
        exit(4)

    # print user choices
    print("\nYour input choices:")
    print("name = {}".format(sys.argv[1]))
    print("max_conf = {}".format(sys.argv[2]))
    print("video to analyse = {}".format(sys.argv[3]))

    # prepare for training
    name = sys.argv[1]
    training_path = "./training-set/{}".format(sys.argv[1])
    max_confidence = float(sys.argv[2])
    video_path = "./videos-input/{}".format(sys.argv[3])

    image_paths = [os.path.join(training_path, f) for f in os.listdir(training_path)]

    # commence training
    print("\nCommencing training for{}..".format(name))
    train(image_paths)

    # try to recognize the person's face in the video file
    print("\nAttempting to analyze {} for {}'s face..".format(sys.argv[3], name))
    process_video(video_path, max_confidence, name)


if __name__ == "__main__":
    main()
