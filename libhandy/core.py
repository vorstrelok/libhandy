import cv2
import numpy as np
# controls - draw a rectangle and save hand hist
# substract background by averaging

# future:
# capture hand via fullscreen
# find face and delete it as a background
# convexhull?
# HOG?

# deal with the "shadow" - hand becoming background if standig still
# detect hand, then find its countour
# and do not count it as a background

# useful bg subs
# COLOR_BGR2HLS - MOG2 1st channel
# COLOR_BGR2HSV - MOG2 2nd channel


def hand_histogram(frame, track_window):
    r, h, c, w = track_window
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def substact_background(frame):
    pass


def main():
    camera_feed = cv2.VideoCapture(0)
    # set camera resolution
    # camera_feed.set(3, 1280)
    camera_feed.set(3, 640)
    # camera_feed.set(4, 720)
    camera_feed.set(4, 480)

    ret, frame = camera_feed.read()
    COLORSPACE = cv2.COLOR_BGR2HSV

    bgsubs = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)
    library_name = 'libhandy v0.1'

    # setup initial location of window
    r, h, c, w = 80, 80, 80, 80
    track_window = (c, r, w, h)
    # Setup the termination criteria,
    # either 20 iteration or move by atleast 5 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 5)
    roi_hist = np.zeros((2, 2))
    while True:
        # Capture frame-by-frame
        ret, frame = camera_feed.read()
        # flip the image so the screen acts like a mirror
        cv2.flip(frame, 1, frame)

        if roi_hist.any():
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply camshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
        else:
            img2 = frame

        cv2.rectangle(frame, (r, c), (r+h, c+w), (20, 20, 55), 5)
        # background = bgsubs.apply(frame[:, :, 2], learningRate=0.01)
        # result_frame = np.zeros_like(frame)
        # result_frame = cv2.bitwise_and(frame, frame, result_frame, background)

        cv2.putText(img2, 'Hand postiond is {0},{1}'.format(frame.shape[1], frame.shape[0]), (int(0.08*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(library_name, gray)
        cv2.imshow(library_name, img2)
        pressed_key_code = cv2.waitKey(10) & 0xFF
        if pressed_key_code == ord('q'):
            break
        elif pressed_key_code == ord('b'):
            pass
        elif pressed_key_code == ord('c'):
            roi_hist = hand_histogram(frame, track_window)

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
