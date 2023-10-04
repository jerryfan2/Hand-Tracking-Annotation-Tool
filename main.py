import cv2
import mediapipe as mp
import numpy as np
import math
from tkinter import filedialog
import os
from pdf2image import convert_from_path
from PIL import Image
import pathlib

def pointDistance(p1, p2):
    return math.sqrt(math.fabs(p1[0]-p2[0])**2 + math.fabs(p1[1]-p2[1])**2)

# inverted colour
def draw(canvas, position, last_position, thickness, colour):
    cv2.circle(canvas, position, thickness//2, colour, cv2.FILLED)
    if index_lx and index_ly:
        cv2.line(canvas, last_position, position, colour, thickness)

def checkFingerVerticalUp(finger, threshold):
    thumb_tip = landmark_pts[fingerTipIDs[finger]]
    thumb_mcp = landmark_pts[fingerTipIDs[finger]-2]
    return math.fabs(thumb_tip[0] - thumb_mcp[0]) <= threshold and thumb_tip[1] < thumb_mcp[1]

def checkFingersHorizontal():
    for i in range(1, 5):
        tip = landmark_pts[fingerTipIDs[i]]
        pip = landmark_pts[fingerTipIDs[i]-2]
        if math.fabs(tip[1]-pip[1]) >= 20:
            return False
    return True

def posOnScreen(p):
    return 0 <= p[0] <= vid_width and 0 <= p[1] <= vid_height

def checkFingersDownCondition(req):
    return all(fingersUp[i] == 0 for i in req)

def checkThumbsUp():
    return all(posOnScreen(landmark_pts[f]) and posOnScreen(landmark_pts[f-2]) for f in fingerTipIDs)\
        and checkFingerVerticalUp(0, 20) and fingersUp[0] and checkFingersDownCondition(thumbsUp) and checkFingersHorizontal()

def checkFingersUp(fingersUpList):
    # thumb is special case: compare tip with ip instead of pip
    for i in range(0, 5):
        tip_cy = landmark_pts[fingerTipIDs[i]]
        pip_cy = landmark_pts[fingerTipIDs[i]-2]
        dip_cy = landmark_pts[fingerTipIDs[i]-1]
        hand_palm0 = landmark_pts[0]
        if i == 0 and pointDistance(tip_cy, hand_palm0) > pointDistance(dip_cy, hand_palm0):
            fingersUpList[0] = 1
        elif i != 0 and pointDistance(tip_cy, hand_palm0) > pointDistance(pip_cy, hand_palm0):
        # if distance form finger tip to palm is farther than pip to palm, then finger is up
            fingersUpList[i] = 1

def saveResultsPNG():
    cv2.imwrite("signature.png", imageCanvasNot)
    cv2.imwrite("webcam.png", imageWebcam)
    if pdfMode:
        imageNoPadding = removeImgPadding(image)
        cv2.imwrite("finalResult.png", imageNoPadding)
        imgPIL = Image.fromarray(imageNoPadding)
        imgPIL.save("signature_on_pdf.pdf", "PDF", resolution=100.0, save_all=True)
    else:
        cv2.imwrite("finalResult.png", image)

def removeImgPadding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

# taken from https://stackoverflow.com/a/44659589
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

vid_width, vid_height = 1280, 720

video = cv2.VideoCapture(0)
video.set(3, vid_width)
video.set(4, vid_height)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)

# in BGR:    black        blue           green       red          yellow         orange          purple         pink            cyan        white
colours = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 165, 255), (128, 0, 128), (203, 192, 255), (255, 255, 0), (255, 255, 255)]
curr_colour = 0 # 0 to 9

fingerTipIDs = [4, 8, 12, 16, 20]
pen_thickness = 8
erase_thickness = 30
index_lx, index_ly = None, None

canWrite = False
eraseMode = False

pdfMode = False
imgMode = False

running = True

win_name = "Output"

while running:
    success, imageWebcam = video.read()
    imageWebcam = cv2.flip(imageWebcam, 1)
    imageRGB = cv2.cvtColor(imageWebcam, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if not pdfMode and not imgMode:
        image = imageWebcam
    elif not imgMode:
        image = resizeAndPad(imagePDF, (vid_height, vid_width))
    else:
        image = resizeAndPad(imgNPRGB, (vid_height, vid_width))

    image_h, image_w, c = image.shape

    imageCanvasNot = cv2.bitwise_not(imageCanvas)
    imageCanvasNot[np.all(imageCanvasNot == (0, 0, 0), axis=-1)] = colours[curr_colour]
    image = np.where(imageCanvas == (0, 0, 0), image, imageCanvasNot)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            landmark_pts = []
            fingersUp = [0] * 5

            for landmark_id, landmark in enumerate(handLms.landmark):
                landmark_cx, landmark_cy = int(landmark.x * image_w), int(landmark.y * image_h)
                landmark_pts.append([landmark_cx, landmark_cy])
                
            checkFingersUp(fingersUp)

            index_cx = landmark_pts[fingerTipIDs[1]][0]
            index_cy = landmark_pts[fingerTipIDs[1]][1]
            # draw black circles on canvas
            drawMode = [2, 3, 4] # indexes of fingers that need to be down
            if canWrite and fingersUp[1] and checkFingersDownCondition(drawMode):
                draw(imageCanvas, (index_cx, index_cy), (index_lx, index_ly), pen_thickness, (255, 255, 255))
                index_lx, index_ly = index_cx, index_cy
            elif eraseMode and fingersUp[1] and checkFingersDownCondition(drawMode):
                draw(imageCanvas, (index_cx, index_cy), (index_lx, index_ly), erase_thickness, (0, 0, 0))
                index_lx, index_ly = index_cx, index_cy
            else:
                index_lx, index_ly = None, None

            # detect thumbs up
            thumbsUp = [1, 2, 3, 4]
            if not canWrite and checkThumbsUp():
                saveResultsPNG()
                running = False

            # check pinky up
            clearScreen = [1, 2, 3]
            if not canWrite and checkFingersDownCondition(clearScreen) and checkFingerVerticalUp(4, 50):
                imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)

            cv2.circle(image, (landmark_pts[fingerTipIDs[1]][0], landmark_pts[fingerTipIDs[1]][1]), erase_thickness//2, (0, 255, 255), cv2.FILLED)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 0, 255)),
                                  mpDraw.DrawingSpec(color=(0, 255, 0)))
            
    
    cv2.imshow(win_name, image)
    cv2.moveWindow(win_name, 0, 0)

    keys = cv2.waitKey(1) & 0xFF
    if keys == ord('q'):
        running = False
    elif keys == ord('s'):
        saveResultsPNG()
        running = False
    elif keys == ord('w'):
        canWrite = not canWrite
        if canWrite:
            eraseMode = False
    elif keys == ord('e'):
        eraseMode = not eraseMode
        if eraseMode:
            canWrite = False
    elif keys == ord("c"):
        imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)
    elif keys == ord('r'):
        index_lx, index_ly = None, None
        canWrite = False
        eraseMode = False
        pdfMode = False
        imgMode = False
        running = True
        imageCanvas = np.zeros((vid_height, vid_width, 3), np.uint8)
        pen_thickness = 8
        erase_thickness = 30
        curr_colour = 0
    elif keys == ord(","):
        if canWrite and pen_thickness - 6 >= 4:
            pen_thickness -= 6
        elif eraseMode and erase_thickness - 6 >= 4:
            erase_thickness -= 6
    elif keys == ord("."):
        if canWrite:
            pen_thickness += 6
        elif eraseMode:
            erase_thickness += 6
    elif ord("0") <= keys <= ord("9"):
        curr_colour = int(chr(keys))
    elif keys == ord("o"):
        file = filedialog.askopenfile(mode='r', filetypes=[("pdf file", "*.pdf"), ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file:
            filepath = os.path.abspath(file.name)
            if pathlib.Path(file.name).suffix == ".pdf":
                pages = convert_from_path(filepath)
                imagePDF = np.array(pages[0])
                pdfMode = True
            else:
                imgPic = Image.open(filepath)
                imgNP = np.asarray(imgPic)
                imgNPRGB = cv2.cvtColor(imgNP, cv2.COLOR_BGR2RGB)
                imgMode = True

video.release()
cv2.destroyAllWindows()

# maybe add zoom in, zoom out functionality?