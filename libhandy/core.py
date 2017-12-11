import os
import sys
import cv2
import numpy as np


def hand_histogram(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [90], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def camshift_tracking(image, roi_hist, track_window):
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

    center = (int(ret[0][0]), int(ret[0][1]))
    return track_window, center, cv2.contourArea(pts), img2


def preprocess_image(image):
    bilaterial_image = cv2.bilateralFilter(image, 7, 75, 75)
    return bilaterial_image


def mog2_bg_subtractor():
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, detectShadows=False
    )
    bg_count = 60
    while True:
        image = yield
        if bg_count:
            bg_count -= 1
            processed_image = bg_sub.apply(image)
        else:
            processed_image = bg_sub.apply(image, learningRate=0.0)
        processed_image = cv2.medianBlur(processed_image, 3)
        kernel = np.ones((7, 7), np.uint8)
        processed_image = cv2.morphologyEx(
            processed_image, cv2.MORPH_CLOSE, kernel
        )
        no_bg_image = cv2.bitwise_and(image, image, mask=processed_image)
        yield no_bg_image


def custom_bg_subtractor():
    '''
    Given tagret image and background tries to
    subtract background from image
    '''
    background = np.zeros([])
    while True:
        image = yield
        if not background.any():
            background = image
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
        yield no_bg_image


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_image():
    test_run = len(sys.argv) - 1
    if test_run:
        bg_image_counter = 64
    else:
        camera_feed = cv2.VideoCapture(0)
        # set camera resolution
        # camera_feed.set(3, 1280)
        # camera_feed.set(4, 720)
        camera_feed.set(3, 640)
        camera_feed.set(4, 480)
    while True:
        image = yield
        if test_run:
            if bg_image_counter > 4:
                image = cv2.imread('metrics/image1.png')
            else:
                image = cv2.imread('metrics/image{}.png'.format(bg_image_counter))
            bg_image_counter -= 1
        else:
            image = camera_feed.read()[1]
        yield image


def main():
    library_name = 'libhandy v0.1'

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

    datafeed = get_image()
    mg_model = mog2_bg_subtractor()
    timeout = 25
    # used in writing a screenshot
    screenshot_index = 1
    while True:
        # Capture frame-by-frame
        datafeed.send(None)
        image = datafeed.send(None)

        # flip the image so the screen acts like a mirror
        image = cv2.flip(image, 1)

        mg_model.send(None)
        processed_image = mg_model.send(image)

        # rudimentary user interface
        pressed_key_code = cv2.waitKey(10) & 0xFF
        if pressed_key_code == ord('q'):
            break
        elif pressed_key_code == ord('s'):
            # make a screenshot
            cv2.imwrite('metrics/image{}.png'.format(screenshot_index), image)
            screenshot_index += 1
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
            track_window, center, area, tracked_image = camshift_tracking(
                processed_image, roi_hist, track_window
            )
            event = ''
            if not 2000 < area < 40000:
                event = 'hand lost'
                track_window = (80, 80, 80, 80)
            # actions
            if timeout:
                timeout -= 1
            else:
                if center[0] < 160:
                    event = 'left'
                    timeout = 25
                elif center[0] > 480:
                    event = 'right'
                    timeout = 25
                else:
                    event = 'center'
            message = 'Center coordinates {0},{1}---{2}'.format(center[0], center[1], event)
        else:
            r, h, c, w = track_window
            # draw target rectangle
            cv2.rectangle(processed_image, (r, c), (r+h, c+w), (20, 20, 55), 2)
            message = ''

        # put some text on the screen
        bla = (80, 80, 80, 80)
        print(intersection_over_union(bla, track_window), track_window)
        cv2.putText(
            processed_image,
            message,
            (int(0.08*image.shape[1]), int(0.97*image.shape[0])),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255)
        )

        # Display the resulting frame
        cv2.namedWindow(library_name, cv2.WINDOW_NORMAL)
        cv2.imshow(library_name, processed_image)

    # When everything done, release the capture
    camera_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
