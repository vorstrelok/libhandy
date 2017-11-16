import cv2
import numpy as np
# global variables
background = np.zeros([])
# 1 - выровнять освещенность и уменьшить разнообразие (одинаковые цвета?)
# HLS is not stable to shadows sadly
# HLS - move all colors to say 5 - 15 bins
# бекграунд - hls 2 канала + вычет (или взять среднее?) \\ mog2 + median и на этом успокоиться
# сравнивать текущий кадр и бекграунд, если разница менее то 10% (!!!) то это бекграунд
# написать в дипломе, что задача вычитания бекграунда неотделима от задачи детекции
# вроде сделали
# 2 - заставить работать с этим camshift
# нарисовать квадратик и сохранить гистограмму руки
# 3 - управление сделать слева и справа экрана две области пересечения
# 4 - протестировать дома презентацию и изменение dpi через переходник
# 5 - сделать HOG

# HOG:
# сделать сетку из кружочков или квадратиков и сравнивать с информацией о руке. какой ближе всего там и рука.
# можно использовать информацию из предыдущего кадра и начинать искать там, где была рука.
# HOG пирамида для изменения дистанции до руки?

# future:
# Feature Matching + Homography to find Objects
# capture hand via fullscreen
# convexhull?
# find face and delete it as a background
# deal with the "shadow" - hand becoming background if standig still
# even lighting conditions - HLS + CLAHE - sorta not working

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


def camshift_tacking(image):
    r, h, c, w = 80, 80, 80, 80
    track_window = (c, r, w, h)
    # Setup the termination criteria,
    # either 20 iteration or move by atleast 5 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 5)
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


def preprocess_image(image):
    # gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)
    bilaterial_image = cv2.bilateralFilter(image, 7, 75, 75)
    # hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahed_lighting = clahe.apply(hls_image[:, :, 1])
    # hls_image[:, :, 1] = clahed_lighting
    # hls_image[:, :, 0] = cv2.bilateralFilter(hls_image[:, :, 0], 7, 75, 75)
    # hls_image[:, :, 2] = cv2.bilateralFilter(hls_image[:, :, 2], 7, 75, 75)
    # bgr_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    # gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    return bilaterial_image


def subtract_background(image, background):
    '''
    Given tagret image and background tries to
    subtract background from image
    '''
    # apply preprocessing
    prepro_image = preprocess_image(image)
    # lets move image and background to hls colorspace
    # and calculate % difference in H and S channels
    prepro_hls = cv2.cvtColor(prepro_image, cv2.COLOR_BGR2HLS)
    background_hls = cv2.cvtColor(background, cv2.COLOR_BGR2HLS)
    subs_h = cv2.absdiff(prepro_hls[:, :, 0], background_hls[:, :, 0])
    subs_s = cv2.absdiff(prepro_hls[:, :, 2], background_hls[:, :, 2])
    diff_h = cv2.divide(subs_h, background_hls[:, :, 0] + 1)
    diff_s = cv2.divide(subs_s, background_hls[:, :, 2] + 1)
    diff = cv2.add(diff_h, diff_s)
    # if the % difference is big enough it is a background
    _, thresholded = cv2.threshold(diff, 0.55, 255, cv2.THRESH_BINARY)
    # aplly median filter and opening morphology
    # to lower noise and obtain a background mask
    median = cv2.medianBlur(thresholded, 7)
    kernel = np.ones((9, 9), np.uint8)
    morphed = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    # finally, apply mask to image
    no_bg_image = cv2.bitwise_and(image, image, mask=morphed)
    return no_bg_image


def apply_filters(image):
    global background
    if background.any():
        processed_image = subtract_background(image, background)
    else:
        background = preprocess_image(image)
        processed_image = image
    return processed_image


def main():
    camera_feed = cv2.VideoCapture(0)
    # set camera resolution
    # camera_feed.set(3, 1280)
    # camera_feed.set(4, 720)
    camera_feed.set(3, 640)
    camera_feed.set(4, 480)
    library_name = 'libhandy v0.1'
    # setup initial location of window
    roi_hist = np.zeros((2, 2))
    while True:
        # Capture frame-by-frame
        _, frame = camera_feed.read()
        # flip the image so the screen acts like a mirror
        frame = cv2.flip(frame, 1)
        frame = apply_filters(frame)
        # draw target rectangle
        # cv2.rectangle(frame, (r, c), (r+h, c+w), (20, 20, 55), 5)
        # put some text on the screen
        # cv2.putText(frame, 'Hand postiond is {0},{1}'.format(frame.shape[1], frame.shape[0]), (int(0.08*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))
        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        cv2.imshow(library_name, frame)

        pressed_key_code = cv2.waitKey(10) & 0xFF
        if pressed_key_code == ord('q'):
            break
        elif pressed_key_code == ord('c'):
            roi_hist = hand_histogram(frame, track_window)

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
