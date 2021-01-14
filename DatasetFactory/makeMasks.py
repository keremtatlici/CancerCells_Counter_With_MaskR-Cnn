import cv2
import numpy as np 
import bench

maskPath = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/Archives/archive 3/Masks/masks.npy'
maskFuturePath = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/masks/'

images = np.load(maskPath,mmap_mode='r')

numberOfImages = images.shape[0]

def makeMasks(image):
    size = image.shape[:2]

    finalImage = np.zeros(size,dtype=np.uint8)

    counter = 1
    for idx in range(0,5):
        img = image[:,:,idx]
        img = img.astype('uint8')
        imgColorList = np.unique(img)

        for coloridx in range(1,len(imgColorList)):
            color = imgColorList[coloridx]
            img = np.where(img != color,img,counter)
            counter+=1

        #resimdeki alan boş ise yani 0 rengine sahipse maskeleri yapıştır.Ama dolu olan kısma bir şey yapıştırma
        finalImage = np.where(finalImage != 0 , finalImage,img)
    return finalImage


for imageidx in range(0,numberOfImages):
    image = images[imageidx,:,:,:]
    image = makeMasks(image)
    savePath = maskFuturePath+str(imageidx+5179)+'.tiff'
    print(savePath)
    cv2.imwrite(savePath,image)
