""" Handles point descriptors"""
import cv2
import numpy as np

inVidPath = "files/4.mp4"
outVidPath = "files/4out.mp4"
outCsvPath = "files/4.csv"

class KPoint:
    '''
    self._label
    self._pt
    self._size
    self._desc
    '''
    def __init__(self, label, kp, desc):
        self._label = label
        self._pt = [int(point) for point in kp.pt] #do something?
        self._size = kp.size
        self._desc = desc

    def __eq__(self, kp2):
        return all(self.pt == kp2.pt)

    def __str__(self):
        ret = "kp-%s in point %d,%d, size of %f" % (self._label, self._pt[0], self._pt[1], self._size)
        return ret

last_name = 0
def find_features(frame, old_keyPoints):
    ret = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(50)
    kp, desc = sift.detectAndCompute(gray,None)
    #do some cleaning on kp
    for pt,des in zip(kp,desc):
        label = "AA"#next_name()
        ret.append(KPoint(label,pt,des))

    print(ret[2])
    return ret

def next_name():
    global last_name
    last_name += 1
    return (
        str(chr(ord('A') + last_name / 26**2))
        + str(chr(ord('A') + (last_name / 26) % 26))
        + str(chr(ord('A') + last_name % 26))
    )

def opticalFlow(old_frame, curr_frame, keyPoints):
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    pt, st, er = cv2.calcOpticalFlowPyrLK(
        old_frame,
        curr_frame,
        np.array([[float(point) for point in x._pt] for x in keyPoints]).astype(np.float32),
        None,
        winSize  = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    for kp,p in zip(keyPoints,pt):
        p = p[0]
        kp._pt = [int(point) for point in p]

def export_to_csv(keyPoints):
    pass

def viz_and_export(frame, keyPoints):
    viz_frame = frame.copy()
    for kp in keyPoints:
        a = kp._pt[0]
        b = kp._pt[1]
        cv2.circle(viz_frame,(a,b),5,(255,0,0),-1)
    cv2.imshow('viz',viz_frame)

def img_process(curr_frame):
    return curr_frame[:-75][:]


def main():
    kp = []
    old_frame = None
    cap = cv2.VideoCapture(inVidPath)   #load video
    for i in range(25*45):
        cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()         #read next frame
        frame = img_process(frame);

        print(old_frame)
        if old_frame is not None:
            opticalFlow(old_frame,frame,kp)
            print("here")
        else:
            kp = find_features(frame,kp)

        viz_and_export(frame,kp)


        #cv2.imshow('frame',gray)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        old_frame = frame

    cap.release()
    cv2.destroyAllWindows()




'''
    img = cv2.imread('pic1.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    kp2 = sift.detect(gray,None)
    kp22,des2 = sift.compute(gray,[kp2[0]])
    img2 = img.copy()
    img3 = img.copy()
    corners = cv2.goodFeaturesToTrack(gray,1600,0.0002,0,0)
    print len(corners)
    corners = np.int0(corners)
    for i in corners:

        x,y = i.ravel()
        cv2.circle(img3,(x,y),3,255,-1)

    img2 = cv2.drawKeypoints(gray,kp,img2)
    cv2.imshow("maz",img2)
    cv2.imshow("maz2",img3)
    print len(kp)
    #for i in range(0,len(kp)):
    #    print kp[i].size
    #print des[0]
    #print des2[0]
    cv2.waitKey(0)
'''

if __name__ == "__main__":
    main()
