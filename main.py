import time
import datetime
import numpy as np
import cv2 as cv
import argparse


def checkTime(now_time, old_time, t=60):
    """
    Check if the time passed t seconds
    :return:
    """
    return now_time - old_time >= t


def startRecording(output, frame, time_not_occ, time_occ):
    """
    Record a video when the camera detects something
    :return:
    """
    global counter, recording_start

    output.write(frame)
    if checkTime(time_not_occ, time_occ, t=5):
        counter += 1
        recording_start = False
        output.release()
        return  # ending the thread with return



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-r", "--record", help="no recording")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
# otherwise, we are reading from a video file
else:
    cap = cv.VideoCapture(args["video"])

# get frame width, height and fps
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)

# recording variables
counter = 0
old_time = time.time()
recording_start = False
output = ''
time_occ = 0

# instantiate background subtraction GSoC
background_subtr_method = cv.bgsegm.createBackgroundSubtractorGSOC()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text that appears on the output
    text = 'Unoccupied'
    time_not_occ = time.time()

    # pass the frame to the background subtractor
    foreground_mask = background_subtr_method.apply(frame)
    # obtain the background without foreground mask
    background_img = background_subtr_method.getBackgroundImage()

    # create bounding boxes for moving objects
    contours, hierarchy = cv.findContours(foreground_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)  # determine the area of objects
        if area > 1200:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

            text = 'Occupied'
            time_occ = time.time()

    # start recording
    if args.get("record", None) is None:
        if checkTime(time.time(), old_time, t=5) is True:
            if text == 'Occupied' and recording_start is False:
                vid_cod = cv.VideoWriter_fourcc(*'mp4v')
                filename = "cam_recording_" + str(datetime.datetime.now()).replace(' ', '_')
                output = cv.VideoWriter("Videos/" + filename + ".mp4", vid_cod, fps, (int(width), int(height)))
                recording_start = True

            if recording_start is True and output != '':
                startRecording(output, frame, time_not_occ, time_occ)


    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, f'[+] Room Status: {text}', (10, 20), font, 0.5, (0, 255, 0), 2)    # (10,20) is the point where text appears, 2 is thickness
    cv.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
               (10, frame.shape[0]-10), font, 0.5, (0, 255, 0), 1)                     # 0.35 is text size, frame.shape[0] - 10 send it to the bottom


    cv.imshow("Initial Frames", frame)
    cv.imshow("Foreground Masks", foreground_mask)
    cv.imshow("Subtraction Result", background_img)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
