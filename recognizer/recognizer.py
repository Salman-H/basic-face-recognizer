"""
module for recognizing a certain face in video frames using a
recognizer that is been trained on that face

AUTHOR:
    Salman Hashmi
    sah517@g.harvard.edu
"""
import cv2
import numpy as np

# NOTE: Region Of Interest or ROI for the purposes of this module is a face

# arbitrary label for training image ROIs
LABEL = 1

# load LBP cascade for detecting faces in video frames
FACE_CLASSIFIER = cv2.CascadeClassifier("./classifiers/lbpcascade_frontalface.xml")

# Create the face recognizer object
LBPH_RECOGNIZER = cv2.createLBPHFaceRecognizer()


def train(paths):
    """
    trains LBPH recognizer on images of a person
    :param paths: paths of all images in the training set
    :return: nothing
    """
    # to contain roi of images and labels for recognizer
    roi_images = []
    labels = []

    # configuring the face images for training
    for image_path in paths:

        # Read the image
        color_image = cv2.imread(image_path)
        training_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in input image and returns as a list of rectangles
        # Refer to OpenCV documentation of detectMultiScale method for parameter details
        training_faces = FACE_CLASSIFIER.detectMultiScale(
            training_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # rectangles specified by top-left (x,y) position
        for (x, y, width, height) in training_faces:
            # if face detected, append to roi_images
            roi_images.append(training_image[y: y + height, x: x + width])

            # append extracted image label
            labels.append(LABEL)

            cv2.imshow("Adding faces to training set...", training_image[y: y + height, x: x + width])
            cv2.waitKey(150)

    # close any opened windows during image training
    cv2.destroyAllWindows()

    # Train the recognizer with extracted faces and labels
    LBPH_RECOGNIZER.train(roi_images, np.array(labels))


def process_video(path, max_conf, name):
    """
    processes a video specified by path
    :param path: path to the video file
    :param max_conf: all conf values below this would indicate face recognition
    :param name: name of the person whose face is to be recognized
    :return: nothing
    """

    # capture frames in the video and configure video output
    capture = cv2.VideoCapture(path)

    # 4-character code of codec used to compress the frames (XVID is compatible with AVI formats)
    fourcc = cv2.cv.CV_FOURCC('x', 'v', 'i', 'd')

    # frames per second size of the output video container
    fps = 17.0
    size = (int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    # N.B. If output format is modified then corresponding fourcc codec must be found and replaced above
    video_out = cv2.VideoWriter('./videos-output/output-' + str(name) + '.avi', fourcc, fps, size)

    while not capture.isOpened():
        capture = cv2.VideoCapture(path)
        cv2.waitKey(100)
        print("Wait for the header")

    # extract current frame from video capture
    frame_position = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

    # read each frame until end of file
    while True:
        flag, color_frame = capture.read()

        if flag:
            # frame successfully captured and ready

            # try to recognize face
            recognize_face(color_frame, max_conf, name)

            # write the current frame
            video_out.write(color_frame)

            # show the video frame after all faces in it have been detecting and recognized
            cv2.imshow('video', color_frame)
            frame_position = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            print(str(frame_position) + " frames")

        else:
            # next frame not ready so try reading it again
            capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_position - 1)
            print("frame is not ready")

            # good to wait a bit before next frame is ready
            cv2.waitKey(500)

        # stop if the number of captured frames is equal to the total number of frames,
        if capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break

    capture.release()
    video_out.release()
    cv2.destroyAllWindows()


# helper for process_video
def recognize_face(color_frame, max_conf, name):
    """
    tries to recognize a detected face in an image or frame using a trained recognizer
    :param color_frame: the current frame read with color
    :param max_conf: the max confidence allowed for recognizing a face
    :param name: the name of the person the face belongs to
    :return: nothing
    """

    # The frame is ready and already captured
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # Detects ROIs/faces in the current gray-scale frame and returns as a list of rectangles
    # minSize is the minimum possible object size i.e. smaller objects are ignored.
    faces = FACE_CLASSIFIER.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # For each detected face in the current video frame
    for (x, y, width, height) in faces:

        # predict the person this face belongs to and get the confidence of this prediction
        nbr_predicted, conf = LBPH_RECOGNIZER.predict(gray_frame[y: y + height, x: x + width])
        # print("\nbr_predicted, conf: {}, {}".format(nbr_predicted, conf))

        if nbr_predicted == LABEL and conf < max_conf:
            print("Recognized face")

            # mark this face with a green rectangle and label with name of person
            mark_face(color_frame, (x, y), (width, height), (0, 255, 0), name)

        else:
            print("No face recognized")

            # mark this face as unknown with a red rectangle
            mark_face(color_frame, (x, y), (width, height), (0, 0, 255), 'unknown')


# helper for recognize_face
# Pycharm editor seems to be ok with following syntax tuples
def mark_face(frame, (x, y), (width, height), (b, g, r), text):
    """
    marks a face with text and a colored rectangle specified by top-left (x,y) position
    :param frame: current frame containing the face to be marked
    :param text: to label the face
    :return: nothing
    """

    # Draw a rectangle around the recognized face colored with specified RGB value
    cv2.rectangle(frame, (x, y), (x + width, y + height), (b, g, r), 2)

    # label the rectangle around the face with the person's name
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x, y + height + 20), font, 0.5, (b, g, r), 2)