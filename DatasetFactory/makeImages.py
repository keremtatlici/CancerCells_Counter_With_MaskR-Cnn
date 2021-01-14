import cv2
import numpy as np
import bench

imagePath = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/Archives/archive 3/Images/images.npy'
imageFuturePath = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/images/'

images = np.load(imagePath, mmap_mode='r')

numberOfImages = images.shape[0]

for imageidx in range(0,numberOfImages):
    image = images[imageidx,:,:,:]
    image = image.astype('uint8')
    savePath = imageFuturePath+str(imageidx)+'.tiff'
    print(savePath)
    cv2.imwrite(savePath,image)