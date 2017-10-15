import cv2
import numpy as np


def main():
    camera_feed = cv2.VideoCapture(0)

    ret, frame = camera_feed.read()

    avg1 = np.float32(frame)
    avg2 = np.float32(frame)
    library_name = 'libhandy v0.1'

    while(True):
        # Capture frame-by-frame
        ret, frame = camera_feed.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.accumulateWeighted(frame, avg1, 0.1)

        res1 = cv2.convertScaleAbs(avg1)

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(library_name, gray)
        cv2.imshow(library_name, res1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
