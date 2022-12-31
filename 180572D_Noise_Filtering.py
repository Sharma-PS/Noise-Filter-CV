### Implement python code for the following filters
# 1.Mean filter
# 2.Median filter
# 3.Mid-point filter

# Filtering function should implement an NxN kernel where N=3 should be the default.  Use image wrapping for the edge pixels
# Programme should automatically read all JPEG files in the home directory and produce an output with the filter name appended and on the same directory 

#### Import
import numpy as np
import cv2
import os

#### Global var
HOME_DIR = "home"
kernelSize = input("Kernel size (default 3) : ")
if kernelSize == "":
    kernelSize = 3
else:
    kernelSize = int(kernelSize)

#### Open all images
def load_images_from_folder(folder : str):
    images = []
    imagesName = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imagesName.append(filename.split(".")[0])
    return images, imagesName

images, imagesName = load_images_from_folder(HOME_DIR)

#### Wrapping image
def wrappingImage(img, kernelSize : int):
    ### width of the mask
    w = kernelSize // 2

    # Fetch First Rows of for wrapping
    fetchFirstRows = img[0 : w , :]
    fetchLastRows = img[-w : , : ]
    
    imgWrapped = img.copy()
    imgWrapped = np.insert(imgWrapped, 0, fetchLastRows, axis=0)
    imgWrapped = np.append(imgWrapped, fetchFirstRows, axis = 0)
    
    ### Fetch First and Last Colsfro wrapping
    fetchFirstCols = imgWrapped[:, 0 : w]
    fetchLastCols = imgWrapped[:, -w : ]
    imgWrapped = np.concatenate([fetchLastCols,imgWrapped], axis=1)
    imgWrapped = np.append(imgWrapped, fetchFirstCols, axis = 1)

    return imgWrapped

#### Mean Filter
def meanFilter(orginalImg, wrappedImage, kernelSize : int):
    filteredImage = np.zeros(orginalImg.shape,dtype=np.int32)
    image_h, image_w = orginalImg.shape[0], orginalImg.shape[1]

    w = kernelSize//2

    for i in range(w, image_h - w): ## traverse image row
        for j in range(w, image_w - w):  ## traverse image col 
            total = [0,0,0]
            for m in range(kernelSize):
                for n in range(kernelSize):
                    total += wrappedImage[i-w+m][j-w+n]
            filteredImage[i-w][j-w] = total // (kernelSize * kernelSize)
    return filteredImage

#### Meadian Filter
def medianFilter(orginalImg, wrappedImage, kernelSize : int):
    filteredImage = np.zeros(orginalImg.shape,dtype=np.int32)
    image_h, image_w = orginalImg.shape[0], orginalImg.shape[1]

    w = kernelSize//2

    for i in range(w, image_h - w): ## traverse image row
        for j in range(w, image_w - w):  ## traverse image col 

            overlapImg = wrappedImage[i-w : i+w+1, j-w : j+w+1 ]    # Crop image for mask product         
            filteredImage[i][j] = np.median(overlapImg.reshape(-1, 3), axis=0)  # Filtering
            
    return filteredImage

#### Mid Point Filter
def midPointFilter(orginalImg, wrappedImage, kernelSize : int):
    filteredImage = np.zeros(orginalImg.shape,dtype=np.int32)
    image_h, image_w = orginalImg.shape[0], orginalImg.shape[1]

    w = kernelSize//2

    for i in range(w, image_h - w): ## traverse image row
        for j in range(w, image_w - w):  ## traverse image col 

            overlapImg = wrappedImage[i-w : i+w+1, j-w : j+w+1 ]    # Crop image for mask product         
            # change the dtype to 'int32' for add purpose
            overlapImg = overlapImg.astype('int32')   
            maxVal = np.max(overlapImg.reshape(-1, 3), axis=0)
            minVal = np.min(overlapImg.reshape(-1, 3), axis=0)
            filteredImage[i][j] = np.add(maxVal, minVal) // 2
            
    return filteredImage

#### Save Image
def saveImage(fileName : str, img):
    path = HOME_DIR + "/" + fileName + ".jpg"
    cv2.imwrite(path, img)
    print(fileName + " filtered successfully ...")


## iterate all images to apply filter
for idx, image in enumerate(images):
    ### Wrap the image
    wrappedImg = wrappingImage(image, kernelSize)
    ## mean filter image
    meanFilteredImg = meanFilter(image, wrappedImg, kernelSize)
    saveImage(imagesName[idx] + "meanFilter", meanFilteredImg)
    ## median filter image
    medianFilteredImg = medianFilter(image, wrappedImg, kernelSize)
    saveImage(imagesName[idx] + "medianFilter", medianFilteredImg)
    ## mean filter image
    midPointFilteredImg = midPointFilter(image, wrappedImg, kernelSize)
    saveImage(imagesName[idx] + "midPointFilter", midPointFilteredImg)

print("DONE ...")