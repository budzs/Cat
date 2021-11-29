import cv2
import time
import numpy as np
from pyasn1.compat.octets import null

pic = 0
cap = cv2.VideoCapture("D:\\PycharmProjects\\pythonProject2\\files\\3.mp4")
if not cap.isOpened():
    print("Error opening video stream or file")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

try:
    catface_cascade = cv2.CascadeClassifier('visionary.net_cat_cascade_web_LBP.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")

noeyes = False


def adjust_gamma(image, gamma=1.5):
    """
    Use lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    :param image: input image
    :param gamma: gamma value
    :return:
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def automatic_thresh(picture):
    """
    Automatic threshold value calculation
    :param picture: picture
    :return: lower and upper threshold value
    """
    med = np.median(picture)
    sigma = 0.50
    lower_thresh = int(max(0, (1.0 - sigma) * med))
    upper_thresh = int(min(255, (1.0 + sigma) * med))
    return lower_thresh, upper_thresh


def parameters(detector_params):
    """
    Blob parameters
    :param detector_params: created SimpleBlob parameter
    :return: specific filters and values for the parameters
    """
    detector_params.filterByArea = True
    detector_params.maxArea = 50000
    detector_params.minArea = (ey + eh) / 20
    detector_params.minThreshold = 0
    detector_params.maxThreshold = 255
    detector_params.filterByConvexity = True
    detector_params.minConvexity = 0.80
    detector_params.filterByCircularity = True
    detector_params.minCircularity = 0.4
    detector_params.filterByColor = True
    detector_params.blobColor = 0
    return detector_params


def modify(img):
    """
    Morphological modifications
    :param img: input image
    :return: keypoint
    """
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    return detector.detect(img)


left, right = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    try:
        frame = adjust_gamma(frame)
    except:
        pass
    if ret:
        right_eye = null
        left_eye = null
        # frame = cv2.resize(frame, (int(width / 1), int(height / 1)))
        pic += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cat_faces = catface_cascade.detectMultiScale(gray, scaleFactor=1.18, minNeighbors=4, minSize=(100, 100))

        if len(cat_faces) == 0:
            noeyes = True
        for (i, (x, y, w, h)) in enumerate(cat_faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            eye_gray = gray[y:y + h, x:x + w]
            # eye_color = frame[y:y + h, x:x + w]

            height_face = y + h / 4
            eyes = eye_cascade.detectMultiScale(eye_gray, scaleFactor=1.01, minNeighbors=5, maxSize=(120, 120),
                                                minSize=(60, 60))
            for (ex, ey, ew, eh) in eyes:
                if ey > h / 5:
                    break
                else:
                    eye_center = ex + ew / 2
                    detector_params = cv2.SimpleBlobDetector_Params()
                    detector = cv2.SimpleBlobDetector_create(parameters(detector_params))
                    if eye_center > 100:
                        left_eye = eye_gray[ey:ey + eh, ex:ex + ew]
                        threshold_min, threshold_max = automatic_thresh(left_eye)
                        _, l_img = cv2.threshold(left_eye, threshold_min, threshold_max, cv2.THRESH_BINARY)
                        keypoints = modify(l_img)
                        if len(keypoints) > 0:
                            pts = cv2.KeyPoint_convert(keypoints)
                            left = (eye_center - ex) - pts[0][0]
                        else:
                            noeyes = True

                        cv2.drawKeypoints(l_img, keypoints, l_img, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    else:
                        right_eye = eye_gray[ey:ey + eh, ex:ex + ew]
                        threshold_min, threshold_max = automatic_thresh(right_eye)
                        _, r_img = cv2.threshold(right_eye, threshold_min, threshold_max, cv2.THRESH_BINARY)
                        keypoints = modify(r_img)
                        if len(keypoints) > 0:
                            pts = cv2.KeyPoint_convert(keypoints)
                            right = (eye_center - ex) - pts[0][0]
                        else:
                            noeyes = True
                        cv2.drawKeypoints(r_img, keypoints, r_img, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    he, wi = frame.shape[:2]
                    if (left + right) < -3:
                        cv2.rectangle(frame, (int(wi / 2), 0), (wi, he), (255, 255, 0), 5)
                    elif (left + right) > 3:
                        cv2.rectangle(frame, (0, 0), (int(wi / 2), he), (255, 255, 0), 5)
                    else:
                        cv2.rectangle(frame, (int(wi / 4), 0), (int(wi * 3 / 4), he), (255, 255, 0), 5)

        if not noeyes:
            # cv2.imshow('frame', frame)
            cv2.imwrite("D:\\PycharmProjects\\pythonProject2\\files\\cat3\\" + str(pic) + ".png", frame)
        noeyes = False

        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
