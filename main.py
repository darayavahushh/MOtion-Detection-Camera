import numpy as np
import cv2 as cv
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
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

# instantiate background subtraction SuBSENSE
#background_subtr_method = bgs.SuBSENSE()  <- not working yet; needs configuration to run
# instantiate background subtraction GSoC
background_subtr_method = cv.bgsegm.createBackgroundSubtractorGSOC()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # pass the frame to the background subtractor
    foreground_mask = background_subtr_method.apply(frame)
    # obtain the background without foreground mask
    background_img = background_subtr_method.getBackgroundImage()

    # create bounding boxes for moving objects
    contours, hierarchy = cv.findContours(foreground_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)  # determine the area of objects
        if area > 1000:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)


    cv.imshow("Initial Frames", frame)
    cv.imshow("Foreground Masks", foreground_mask)
    cv.imshow("Subtraction Result", background_img)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
