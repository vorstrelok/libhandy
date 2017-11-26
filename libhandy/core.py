import cv2
import numpy as np
import os


def hand_histogram(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def camshift_tacking(image, roi_hist, track_window):
    # Setup the termination criteria,
    # either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # apply camshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(image, [pts], True, 255, 2)

    return track_window, img2


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


def main():
    camera_feed = cv2.VideoCapture(0)
    # set camera resolution
    # camera_feed.set(3, 1280)
    # camera_feed.set(4, 720)
    camera_feed.set(3, 640)
    camera_feed.set(4, 480)
    library_name = 'libhandy v0.1'

    background = np.zeros([])
    # initial location of tracking window
    track_window = (80, 80, 80, 80)
    # if the return value from imread is not None,
    # it means that the read was sucesseful
    # its done this way because imread function does not raise
    # FileNotFoundError if the file does not exist
    hand_image = cv2.imread('hand_img.png')
    if hand_image is not None:
        roi_hist = hand_histogram(hand_image)
    else:
        roi_hist = np.zeros([])

    while True:
        # Capture frame-by-frame
        _, image = camera_feed.read()

        # flip the image so the screen acts like a mirror
        image = cv2.flip(image, 1)

        if background.any():
            processed_image = subtract_background(image, background)
        else:
            background = preprocess_image(image)
            processed_image = image

        # rudimentary user interface
        pressed_key_code = cv2.waitKey(10) & 0xFF
        if pressed_key_code == ord('q'):
            break
        elif pressed_key_code == ord('c'):
            if roi_hist.any():
                # if we already have a hand image and we press C again,
                # threat it as resetting
                roi_hist = np.zeros([])
                track_window = (80, 80, 80, 80)
                os.remove('hand_img.png')
            else:
                # set up the ROI for tracking
                r, h, c, w = track_window
                roi = processed_image[r:r+h, c:c+w]
                roi_hist = hand_histogram(roi)
                # save hand picture for later use
                cv2.imwrite('hand_img.png', roi)

        if roi_hist.any():
            track_window, tracked_image = camshift_tacking(
                processed_image, roi_hist, track_window
            )
        else:
            r, h, c, w = track_window
            # draw target rectangle
            cv2.rectangle(processed_image, (r, c), (r+h, c+w), (20, 20, 55), 2)

        # put some text on the screen
        # cv2.putText(
        #     image,
        #     'Hand postiond is {0},{1}'.format(image.shape[1], image.shape[0]),
        #     (int(0.08*image.shape[1]), int(0.97*image.shape[0])),
        #     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255)
        # )

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        cv2.imshow(library_name, processed_image)

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
