import cv2
import numpy as np
import urllib.request
import  utils


# URL = "abc"

#####################
webcam = True
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,640)
cap.set(4,480)
path = 'photo (3).jpg'
sc =2
wp = 210 * sc
hp = 297 * sc

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',-1, fourcc, 20.0, (640, 480))


while True:
    if webcam:
        # img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
        # img = cv2.imdecode(img_arr, -1)
        success, img = cap.read()
    else: img = cv2.imread(path)
    imgCnt, finalCnt =  utils.getContours(img,
                                       min_area=50000,
                                       filter=4)

    if len(finalCnt) !=0:
        biggest = finalCnt[0][2]
        #print(biggest)
        imgWrap = utils.warpImg(img, biggest, wp, hp)

        imgCnt2, finalCnt2 = utils.getContours(imgWrap,
                                             min_area=200,
                                             filter=4,draw=False)

        if len(finalCnt)!=0:
            for obj in finalCnt2:
                cv2.polylines(imgCnt2,[obj[2]],True,(0,0,255),2)
                nPoints = utils.reorder(obj[2])
                nW = round((utils.findDis(nPoints[0][0] // sc, nPoints[1][0] // sc) / 10), 1)
                nH = round((utils.findDis(nPoints[0][0] // sc, nPoints[2][0] // sc) / 10), 1)
                cv2.arrowedLine(imgCnt2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgCnt2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgCnt2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgCnt2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)


        cv2.imshow('A4', imgCnt2)
        out.write(imgCnt2)
    #img = cv2.resize(img,(0,0),None,0.5,0.5)

    cv2.imshow('Original', img)


    if cv2.waitKey(1)==27:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()