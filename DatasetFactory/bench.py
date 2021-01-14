import cv2
import numpy as np
import copy
import os
from math import sqrt

def changePixelColor(image,pixelsColor,givenColor):
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0,h):
        for x in range(0,w):
            print(image[y,x])
            print(image[y,x].shape)
            print(pixelsColor.shape)
            if image[y,x] == pixelsColor:
                image[y,x] = givenColor
    
    return image

def waitKeyAndDestroyWindows():
    #print("----------------------------------")
    #print('waitKeyAndDestroyWindows Methodu BAŞLADI!')
    cv2.waitKey()
    cv2.destroyAllWindows()

    #print('waitKeyAndDestroyWindows Methodu BİTTİ!')
    #print('----------------------------------')

def resmiKucultVeGoster(img,windowName,percent = 23):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim , interpolation= cv2.INTER_AREA)
    cv2.imshow(windowName,resized)


def deleteSmallContoursFromContoursList(contours,Alan=500):
    #print("----------------------------------")
    #print('deleteSmallContoursFromContoursList Methodu BAŞLADI!')
    print('Başlangıç Contour Sayısı = ', len(contours))
    sonlandirmaKriteri = False
    while sonlandirmaKriteri != True:
        sonlandirmaKriteri = True
        for q in range(len(contours)):
            if cv2.contourArea(contours[q]) < Alan:
                print(q,". contouru küçük olduğu için sildim")
                del contours[q]
                sonlandirmaKriteri = False
                break

    print('Kalan Contour Sayısı = ', len(contours))
    #print('deleteSmallContoursFromContoursList Methodu BİTTİ!')
    #print('----------------------------------')
    return contours

def makeImageGrayIfNot(image):
    #print("----------------------------------")
    #print('makeImageGrayIfNot Methodu BAŞLADI!')
    if len(image.shape) ==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #print('makeImageGrayIfNot Methodu BİTTİ!')
    #print('----------------------------------')

    return image

def makeImageBgrIfNot(image):
    #print("----------------------------------")
    #print('makeImageBgrIfNot Methodu BAŞLADI!')
    if len(image.shape) ==2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #print('makeImageBgrIfNot Methodu BİTTİ!')
    #print('----------------------------------')
    return image

##Bu method ona verilen yoldaki tüm alt dosyaların yollarını listeler
def pathListOfFiles(path):
    print("----------------------------------")
    print('pathListOfFiles Methodu BAŞLADI!')
    listOfFiles = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            listOfFiles.append(os.path.join(subdir, file))
    print('pathListOfFiles Methodu BİTTİ!')
    print('----------------------------------')
    return listOfFiles

#normal veya gray image ver fark etmez
#ona verdiğin resimi alır ve her contourun ayrı olduğu bir resim listesi döndürür.
def makeImage4EachContour(image):
    #print("----------------------------------")
    #print('makeImage4EachContour Methodu BAŞLADI!')

    image_contourList = []
    image = makeImageBgrIfNot(image)
    image_gray = makeImageGrayIfNot(image)
    fill_color = [255,255,255]
    mask_value = 255

    _,image_threshold =cv2.threshold(image_gray,254,255,cv2.THRESH_BINARY_INV)

    contours,_=cv2.findContours(image_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = deleteSmallContoursFromContoursList(contours)

    for contour in contours:
        image_copy = copy.deepcopy(image)
        stencil = np.zeros(image_copy.shape[:-1]).astype(np.uint8)
        cv2.fillPoly(stencil,[contour],mask_value)
        sel = stencil != mask_value
        image_copy[sel] = fill_color
        image_contourList.append(image_copy)

    #print('makeImage4EachContour Methodu BİTTİ!')
    #print('----------------------------------')
    return image_contourList

#K değeri kaç parça renk olacağıdır.Düz img yolla
#Not : Her contour için aynı k değeri ile farklı kmeans uygulamak faydalı olabilir.
def kmeansOnImg(img ,K):
    #print("----------------------------------")
    #print('kmeansOnImg Methodu BAŞLADI!')

    img = makeImageBgrIfNot(img)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    #print('kmeansOnImg Methodu BİTTİ!')
    #print('----------------------------------')
    return res2

#image[y,x] contour[x,y] !!!!!!!!
def whiteRate(image,radius,center):
    #print("----------------------------------")
    #print('whiteRate Methodu BAŞLADI!')

    image = makeImageGrayIfNot(image)
    color = 255
    thickness = -1
    #print("image shape 2 değer olmalı !!! ==",image.shape )
    img = np.zeros(image.shape,np.uint8)

    cv2.circle(img,center,radius,color,thickness)
    points = np.transpose(np.where(img==255))

    AreaOfCircle = 0
    white_counter=0
    for q in points:
        try:
            AreaOfCircle+=1
            if image[q[0],q[1]] == 255:
                white_counter+=1
        except :
            print("exception!! at =  ",q )

    #print('whiteRate Methodu BİTTİ!')
    #print('----------------------------------')
    return white_counter/AreaOfCircle

def BlackBgMakeImage4EachContour(image):

    #print("----------------------------------")
    #print('BlackBgMakeImage4EachContour Methodu BAŞLADI!')

    image_contourList = []
    image = makeImageBgrIfNot(image)
    image_gray = makeImageGrayIfNot(image)
    fill_color = [0,0,0]
    mask_value = 255

    _,image_threshold =cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)

    contours,_=cv2.findContours(image_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = deleteSmallContoursFromContoursList(contours)

    for contour in contours:
        image_copy = copy.deepcopy(image)
        stencil = np.zeros(image_copy.shape[:-1]).astype(np.uint8)
        cv2.fillPoly(stencil,[contour],mask_value)
        sel = stencil != mask_value
        image_copy[sel] = fill_color
        image_contourList.append(image_copy)

    #print('BlackBgMakeImage4EachContour Methodu BİTTİ!')
    #print('----------------------------------')
    return image_contourList

def findDistanceOf2Coordinates(c1,c2):
    width =  abs(c1[0]-c2[0])
    height = abs(c1[1]-c2[1])
    distance = sqrt((width**2) + (height**2))
    return distance


def ImgdenMaskeOlustur(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours ,_ = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        
        cni = contours[i]
        cv2.drawContours(img,[cni],-1,(i+1),-1)

    return img
