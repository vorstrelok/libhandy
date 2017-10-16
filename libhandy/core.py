import cv2
import numpy as np


def main():
    camera_feed = cv2.VideoCapture(0)

    ret, frame = camera_feed.read()
    COLORSPACE = cv2.COLOR_BGR2HSV

    # avg = np.float32(frame)
    # avg = cv2.cvtColor(np.float32(frame), COLORSPACE)
    bgsubs = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)
    avg = frame
    library_name = 'libhandy v0.1'

    while True:
        # Capture frame-by-frame
        ret, frame = camera_feed.read()

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, COLORSPACE)

        # cv2.accumulateWeighted(frame, avg, 0.4)

        # result_frame = frame - cv2.convertScaleAbs(avg)
        # result_frame = frame - avg
        result_frame = bgsubs.apply(frame, learningRate=0.01)
        # kernel = np.ones((3, 3), np.uint8)
        # fg_mask = cv2.erode(result_frame, kernel, iterations=1)
        # result_frame = cv2.bitwise_and(frame, result_frame, mask=fg_mask)

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(library_name, gray)
        cv2.imshow(library_name, result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
