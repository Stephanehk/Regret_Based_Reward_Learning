from mss import mss
import cv2
import time
import sys

# time.sleep(2)
# with mss() as sct:
#     filename = sct.shot()

# img = cv2.imread(filename)

def screenshot():
    print (False)
    with mss() as sct:
        filename = sct.shot()
    img = cv2.imread(filename)
    img = cv2.resize(img,(900,600))
    img = img[100:540, 250:860]
    # cv2.imshow("cropped", img)
    # cv2.waitKey(0)
    cv2.imwrite(sys.argv[1],img)
    print (True)
    sys.stdout.flush()
