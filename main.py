# 121A2012 Gitesh Devlekar
# Import libraries
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from matplotlib import pyplot as plt

# Load image

image2 = Image.open('dataset/raw/Img2.png')

image6 = Image.open('dataset/raw/Img6.png')


# Load reference images
rimage2 = Image.open('dataset/reference/RImg2.png')

rimage6 = Image.open('dataset/reference/RImg6.png')


def psnr(reference, fused, original):
    R2 = np.amax(reference) ** 2
    MSE = np.sum(np.power(np.subtract(reference, original), 2))
    MSE /= (reference.size[0] * original.size[1])
    PSNR = 10 * np.log10(R2 / MSE)

    print("Reference vs Original-", "MSE: ", MSE, "PSNR:", PSNR)

    R2 = np.amax(reference) ** 2
    MSE = np.sum(np.power(np.subtract(reference, fused), 2))
    MSE /= (reference.size[0] * fused.size[1])
    PSNR = 10 * np.log10(R2 / MSE)
    print("Reference vs Fused   -", "MSE: ", MSE, "PSNR:", PSNR)
    print('')


# print("MSE & PSNR of PCA fused image")
# psnr(rimage1, pcafused1, image1)
#
# print("MSE & PSNR of Average fused image")
# psnr(rimage1, averagefused1, image1)

def plot_histogram(image):
    # Split the R, G and B channels
    imageR, imageG, imageB = image.split()

    # Plot the histigrams
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Histogram of image")
    plt.plot(imageR.histogram(), color='red')
    plt.plot(imageG.histogram(), color='green')
    plt.plot(imageB.histogram(), color='blue')
    plt.show()


def channel_split(image):
    # Split the R, G and B channels
    imageR, imageG, imageB = image.split()
    x, y = image.size
    Rchannel = np.zeros((y, x, 3), dtype="uint8")
    Bchannel = np.zeros((y, x, 3), dtype="uint8")
    Gchannel = np.zeros((y, x, 3), dtype="uint8")
    # Create individual components image
    Rchannel[:, :, 0] = imageR
    Bchannel[:, :, 1] = imageG
    Gchannel[:, :, 2] = imageB
    # Convert array to image
    Rchannel = Image.fromarray(Rchannel)
    Bchannel = Image.fromarray(Bchannel)
    Gchannel = Image.fromarray(Gchannel)

    # Plot R, G and B components
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 4, 2)
    plt.title("Red Component")
    plt.imshow(Rchannel)
    plt.subplot(1, 4, 3)
    plt.title("Green Component")
    plt.imshow(Bchannel)
    plt.subplot(1, 4, 4)
    plt.title("Blue Component")
    plt.imshow(Gchannel)
    plt.show()


def compensate_RB(image, flag):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Get maximum and minimum pixel value
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    # Convert to array
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)

    x, y = image.size

    # Normalizing the pixel value to range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = (imageR[i][j] - minR) / (maxR - minR)
            imageG[i][j] = (imageG[i][j] - minG) / (maxG - minG)
            imageB[i][j] = (imageB[i][j] - minB) / (maxB - minB)

    # Getting the mean of each channel
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Compensate Red and Blue channel
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int(
                    (imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
                imageB[i][j] = int(
                    (imageB[i][j] + (meanG - meanB) * (1 - imageB[i][j]) * imageG[i][j]) * maxB)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j] = int(imageG[i][j] * maxG)

    # Compensate Red channel
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int(
                    (imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j] = int(imageB[i][j] * maxB)
                imageG[i][j] = int(imageG[i][j] * maxG)

    # Create the compensated image
    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = imageR;
    compensateIm[:, :, 1] = imageG;
    compensateIm[:, :, 2] = imageB;

    # Plotting the compensated image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("RB Compensated Image")
    plt.imshow(compensateIm)
    plt.show()
    compensateIm = Image.fromarray(compensateIm)

    return compensateIm


def gray_world(image):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Form a grayscale image
    imagegray = image.convert('L')

    # Convert to array
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)

    x, y = image.size

    # Get mean value of pixels
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)

    # Gray World Algorithm
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = int(imageR[i][j] * meanGray / meanR)
            imageG[i][j] = int(imageG[i][j] * meanGray / meanG)
            imageB[i][j] = int(imageB[i][j] * meanGray / meanB)

    # Create the white balanced image
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR;
    whitebalancedIm[:, :, 1] = imageG;
    whitebalancedIm[:, :, 2] = imageB;

    # Plotting the compensated image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Compensated Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("White Balanced Image")
    plt.imshow(whitebalancedIm)
    plt.show()

    return Image.fromarray(whitebalancedIm)


def hsv_global_equalization(image):
    # Convert to HSV
    hsvimage = image.convert('HSV')

    # Plot HSV Image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("White balanced Image")
    plt.imshow(hsvimage)

    # Splitting the Hue, Saturation and Value Component
    Hue, Saturation, Value = hsvimage.split()
    # Perform Equalization on Value Component
    equalizedValue = ImageOps.equalize(Value, mask=None)

    x, y = image.size
    # Create the equalized Image
    equalizedIm = np.zeros((y, x, 3), dtype="uint8")
    equalizedIm[:, :, 0] = Hue;
    equalizedIm[:, :, 1] = Saturation;
    equalizedIm[:, :, 2] = equalizedValue;

    # Convert the array to image
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    # Convert to RGB
    rgbimage = hsvimage.convert('RGB')

    # Plot equalized image
    plt.subplot(1, 2, 2)
    plt.title("Contrast enhanced Image")
    plt.imshow(rgbimage)

    return rgbimage


def sharpen(wbimage, original):
    # First find the smoothed image using Gaussian filter
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)

    # Split the smoothed image into R, G and B channel
    smoothedr, smoothedg, smoothedb = smoothed_image.split()

    # Split the input image
    imager, imageg, imageb = wbimage.split()

    # Convert image to array
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)

    x, y = wbimage.size

    # Perform unsharp masking
    for i in range(y):
        for j in range(x):
            imageR[i][j] = 2 * imageR[i][j] - smoothedR[i][j]
            imageG[i][j] = 2 * imageG[i][j] - smoothedG[i][j]
            imageB[i][j] = 2 * imageB[i][j] - smoothedB[i][j]

    # Create sharpened image
    sharpenIm = np.zeros((y, x, 3), dtype="uint8")
    sharpenIm[:, :, 0] = imageR;
    sharpenIm[:, :, 1] = imageG;
    sharpenIm[:, :, 2] = imageB;

    # Plotting the sharpened image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.subplot(1, 3, 2)
    plt.title("White Balanced Image")
    plt.imshow(wbimage)
    plt.subplot(1, 3, 3)
    plt.title("Sharpened Image")
    plt.imshow(sharpenIm)
    plt.show()

    return Image.fromarray(sharpenIm)


def average_fusion(image1, image2):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convert images to arrays
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1.size

    # Average Fusion
    for i in range(y):
        for j in range(x):
            image1R[i][j] = (image1R[i][j] + image2R[i][j]) / 2
            image1G[i][j] = (image1G[i][j] + image2G[i][j]) / 2
            image1B[i][j] = (image1B[i][j] + image2B[i][j]) / 2

    # Create the fused image
    fusedIm = np.zeros((y, x, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R;
    fusedIm[:, :, 1] = image1G;
    fusedIm[:, :, 2] = image1B;

    # Plot the fused image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("Fused Image")
    plt.imshow(fusedIm)
    plt.show()

    return Image.fromarray(fusedIm)


def pca_fusion(image1, image2):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convert to column vector
    image1R = np.array(image1r, np.float64).flatten()
    image1G = np.array(image1g, np.float64).flatten()
    image1B = np.array(image1b, np.float64).flatten()
    image2R = np.array(image2r, np.float64).flatten()
    image2G = np.array(image2g, np.float64).flatten()
    image2B = np.array(image2b, np.float64).flatten()

    # Get mean of each channel
    mean1R = np.mean(image1R)
    mean1G = np.mean(image1G)
    mean1B = np.mean(image1B)
    mean2R = np.mean(image2R)
    mean2G = np.mean(image2G)
    mean2B = np.mean(image2B)

    # Create a 2*N array where each column represents each image channel
    imageR = np.array((image1R, image2R))
    imageG = np.array((image1G, image2G))
    imageB = np.array((image1B, image2B))

    x, y = imageR.shape

    # Subtract the respective mean from each column
    for i in range(y):
        imageR[0][i] -= mean1R
        imageR[1][i] -= mean2R
        imageG[0][i] -= mean1G
        imageG[1][i] -= mean2G
        imageB[0][i] -= mean1B
        imageB[1][i] -= mean2B

    # Find the covariance matrix
    covR = np.cov(imageR)
    covG = np.cov(imageG)
    covB = np.cov(imageB)

    # Find eigen value and eigen vector
    valueR, vectorR = np.linalg.eig(covR)
    valueG, vectorG = np.linalg.eig(covG)
    valueB, vectorB = np.linalg.eig(covB)

    # Find the coefficients for each channel which will act as weight for images
    if (valueR[0] >= valueR[1]):
        coefR = vectorR[:, 0] / sum(vectorR[:, 0])
    else:
        coefR = vectorR[:, 1] / sum(vectorR[:, 1])

    if (valueG[0] >= valueG[1]):
        coefG = vectorG[:, 0] / sum(vectorG[:, 0])
    else:
        coefG = vectorG[:, 1] / sum(vectorG[:, 1])

    if (valueB[0] >= valueB[1]):
        coefB = vectorB[:, 0] / sum(vectorB[:, 0])
    else:
        coefB = vectorB[:, 1] / sum(vectorB[:, 1])

    # Convert to array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1R.shape

    # Calculate the pixel value for the fused image from the coefficients obtained above
    for i in range(x):
        for j in range(y):
            image1R[i][j] = int(coefR[0] * image1R[i][j] + coefR[1] * image2R[i][j])
            image1G[i][j] = int(coefG[0] * image1G[i][j] + coefG[1] * image2G[i][j])
            image1B[i][j] = int(coefB[0] * image1B[i][j] + coefB[1] * image2B[i][j])

    # Create the fused image
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R;
    fusedIm[:, :, 1] = image1G;
    fusedIm[:, :, 2] = image1B;

    # Plot the fused image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Sharpened Image")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Contrast Enhanced Image")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("PCA Fused Image")
    plt.imshow(fusedIm)
    plt.show()

    return Image.fromarray(fusedIm)


plot_histogram(image2)
channel_split(image2)
compensate_RB(image2, 0)
wbimage2 = gray_world(image2)
contrastimg2 = hsv_global_equalization(wbimage2)
sharpenimg2 = sharpen(wbimage2, image2)
averagefused2 = average_fusion(sharpenimg2, contrastimg2)
pcafused2 = pca_fusion(sharpenimg2,contrastimg2)
channel_split(averagefused2)
channel_split(pcafused2)
print("MSE & PSNR of Average fused image 2")
psnr(rimage2, averagefused2, image2)
print("MSE & PSNR of PCA fused image 2")
psnr(rimage2, pcafused2, image2)

plot_histogram(image6)
channel_split(image6)
compensate_RB(image6, 0)
wbimage6 = gray_world(image6)
contrastimg6 = hsv_global_equalization(wbimage6)
sharpenimg6 = sharpen(wbimage6, image6)
averagefused6 = average_fusion(sharpenimg6, contrastimg6)
pcafused6 = pca_fusion(sharpenimg6,contrastimg6)
channel_split(averagefused6)
channel_split(pcafused6)
print("MSE & PSNR of Average fused image 6")
psnr(rimage6, averagefused6, image6)
print("MSE & PSNR of PCA fused image 6")
psnr(rimage6, pcafused6, image6)

