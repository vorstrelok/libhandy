import cv2
import numpy as np


def main():
    camera_feed = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = camera_feed.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.namedWindow('libhandy v0.1', cv2.WINDOW_NORMAL)
        cv2.imshow('libhandy v0.1', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
