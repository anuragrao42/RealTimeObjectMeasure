import  cv2
import  numpy as np

def getContours(img,cThr = [100,100],min_area=1000,showCanny = False,filter=0,draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny: cv2.imshow("canny",imgThre)

    contours, hierarchy =  cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCnt = []
    for cnt in contours:
        area =  cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx)==filter:
                    finalCnt.append([len(approx),area, approx,bbox,cnt])
            else:finalCnt.append([len(approx),area, approx,bbox,cnt])
    finalCnt = sorted(finalCnt,key=lambda x:x[1],reverse=True)
    if draw:
        for cnt in finalCnt:
            cv2.drawContours(img,cnt[4],-1,(0,0,255),4)

    return img, finalCnt

def reorder(mypoints):
    print(mypoints.shape)
    mypointsNew = np.zeros_like(mypoints)
    mypoints = mypoints.reshape((4,2))
    add = mypoints.sum(1)
    mypointsNew[0] = mypoints[np.argmin(add)]
    mypointsNew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    mypointsNew[1] = mypoints[np.argmin(diff)]
    mypointsNew[2] = mypoints[np.argmax(diff)]
    return mypointsNew


def warpImg(img, points,w,h,pad =20):
    # print(points)
    points =reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWrap = cv2.warpPerspective(img,matrix,(w,h))
    imgWrap = imgWrap[pad:imgWrap.shape[0]-pad, pad:imgWrap.shape[1]-pad]
    return  imgWrap

def findDis(pt1,pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
