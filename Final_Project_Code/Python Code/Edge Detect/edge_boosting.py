import cv2
import numpy as np
from matplotlib import pyplot as plt
test_case = 3

if(test_case ==0):

    img = cv2.imread('knife1.jpg',0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if(test_case ==1):

    img = cv2.imread('knife2.jpg',0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if(test_case ==2):

    img = cv2.imread('knife2.jpg',0)
    edges = cv2.Canny(img, 100, 200)
    edges_L2 = cv2.Canny(img,100,200, L2gradient = True)
    plt.subplot(121),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges_L2,cmap = 'gray')
    plt.title('EdgeL2 Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if(test_case ==3):
    norm = 0
    width =2
    Thresh = 300
    img = cv2.imread('mangun.jpg',0)
    blank = np.zeros(img.shape)
    edges_L2 = cv2.Canny(img,100,200, L2gradient = True)
    lines = cv2.HoughLines(edges_L2, width, 6.28/180, Thresh)
    #lines = cv2.HoughLinesP(edges_L2, 1, 03.14/180, 90)
    threat = 0
    for i in range(0,lines.shape[0]):
        r = lines[i][0][0]
        t = lines[i][0][1]
        xbar = np.cos(t)
        ybar = np.sin(t)
        X = r*xbar
        Y = r*ybar
        scale = 1000
        loc1 = [X + scale * (-ybar), Y + scale * (xbar)]
        loc2 = [X - scale * (-ybar), Y - scale * (xbar)]
        threat += 1
        plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]])
    #plt.subplot(121),plt.imshow((blank+lines),cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122)\
    danger = threat*Thresh/width
    if(norm == 1):
        ambient = danger
    else:
        ambient = 3150
    print(ambient)
    print(danger/ambient)
    plt.imshow(edges_L2,cmap = 'gray')
    plt.title('EdgeL2 Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if(test_case ==4):
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        norm = 0
        width = 2
        Thresh = 100
        #img = cv2.imread('mangun.jpg', 0)
        blank = np.zeros(img.shape)
        edges_L2 = cv2.Canny(img, 100, 200, L2gradient=True)
        lines = cv2.HoughLines(edges_L2, width, 6.28 / 180, Thresh)
        # lines = cv2.HoughLinesP(edges_L2, 1, 03.14/180, 90)
        threat = 0
        for i in range(0, lines.shape[0]):
            r = lines[i][0][0]
            t = lines[i][0][1]
            xbar = np.cos(t)
            ybar = np.sin(t)
            X = r * xbar
            Y = r * ybar
            scale = 1000
            loc1 = [X + scale * (-ybar), Y + scale * xbar]
            loc2 = [X - scale * (-ybar), Y - scale * xbar]
            threat += 1
            plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]])
        danger = threat * Thresh / width
        if (norm == 1):
            ambient = danger
        else:
            ambient = 3150
        print(ambient)
        print(danger / ambient)

        # Display the resulting frame
        cv2.imshow('frame', edges_L2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
