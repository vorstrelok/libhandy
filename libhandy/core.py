import cv2
import numpy as np
# deal with the "shadow" - hand becoming background if standig still

# оформить в решение:
# нарисовать квадратик и сохранить гистограмму руки
# фотографировать бекграунд по кнопке, сравнивать текущий кадр и бекграунд, если разница менее то 10% то это бекграунд

# сделать:
# выровнять освещенность и уменьшить разнообразие (одинаковые цвета?)
# сделать сетку из кружочков или квадратиков и сравнивать с информацией о руке. какой ближе всего там и рука.
# можно использовать информацию из предыдущего кадра и начинать искать там, где была рука.
# HOG пирамида для изменения дистанции до руки?

# future:
# capture hand via fullscreen
# convexhull?
# HOG?
# find face and delete it as a background

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


def subtact_background(frame):
    pass


def main():
    camera_feed = cv2.VideoCapture(0)
    # set camera resolution
    # camera_feed.set(3, 1280)
    # camera_feed.set(4, 720)
    camera_feed.set(3, 640)
    camera_feed.set(4, 480)

    library_name = 'libhandy v0.1'

    # setup initial location of window
    r, h, c, w = 80, 80, 80, 80
    track_window = (c, r, w, h)
    # Setup the termination criteria,
    # either 20 iteration or move by atleast 5 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 5)
    roi_hist = np.zeros((2, 2))
    fg_mask = np.zeros((2, 2))
    while True:
        # Capture frame-by-frame
        ret, frame = camera_feed.read()
        # flip the image so the screen acts like a mirror
        frame = cv2.flip(frame, 1)

        if fg_mask.any():
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            hsv_bg = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2HLS)
            # subs = cv2.absdiff(frame, fg_mask)

            subs = cv2.absdiff(hsv_frame[:, :, :2], hsv_bg[:, :, :2])
            ret, mask = cv2.threshold(subs, 30, 255, cv2.THRESH_TOZERO)
            mask = cv2.add(mask[:, :, 0], mask[:, :, 1])
            # mask = cv2.bitwise_not(mask)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

            frame = cv2.bitwise_and(frame, frame, mask=mask)

            # frame = cv2.GaussianBlur(mask, (5, 5), 0)
            # frame = cv2.medianBlur(frame, 5)

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

        cv2.putText(img2, 'Hand postiond is {0},{1}'.format(frame.shape[1], frame.shape[0]), (int(0.08*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        cv2.imshow(library_name, img2)
        pressed_key_code = cv2.waitKey(10) & 0xFF
        if pressed_key_code == ord('q'):
            break
        elif pressed_key_code == ord('b'):
            fg_mask = cv2.GaussianBlur(cv2.medianBlur(frame, 5), (5, 5), 0)
            # fg_mask = frame
        elif pressed_key_code == ord('c'):
            roi_hist = hand_histogram(frame, track_window)

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
