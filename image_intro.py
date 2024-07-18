# -*- coding: utf-8 -*-


# Read in original RGB image.
import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import Image

#part 1
#read rgb image and display 
ImageA = io.imread('shed1-small.jpg')
(m,n,o) = ImageA.shape
print("ImageA displayed")
plt.imshow(ImageA)
plt.show()

#part 2
# Extract color channels.
RC = ImageA[:,:,0] # Red channel
GC = ImageA[:,:,1] # Green channel
BC = ImageA[:,:,2] # Blue channel
# Create an all black channel.
allBlack = np.zeros((m, n), dtype=np.uint8)
# Create color versions of the individual color channels.
justRed = np.stack((RC, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, GC, allBlack),axis=2)
justBlue = np.stack((allBlack, allBlack, BC),axis=2)
# Recombine the individual color channels to create the original RGB image again.
recombinedRGBImage = np.stack(( RC, GC, BC),axis=2)

io.imsave('justRed1.jpg' , justRed)
io.imsave('justGreen1.jpg' , justGreen)
io.imsave('justBlue1.jpg', justBlue)

print("red")
plt.imshow(justRed)
plt.show()
print("green")
plt.imshow(justGreen)
plt.show()
print("blue")
plt.imshow(justBlue)
plt.show()

#part 3
AG = 0.299*RC + 0.587*GC + 0.114*BC

print("gray level")

plt.imshow(AG, cmap = "gray")

plt.show()
io.imsave('gray_image.jpg', AG.astype(np.uint8))
#part 4
#compute the histogram
histgray = np.zeros(256, dtype=np.intc)
histred=np.zeros(256, dtype = np.intc)
histblue=np.zeros(256, dtype = np.intc)
histgreen=np.zeros(256, dtype = np.intc)

for i in range(m) :
    for j in range(n) :
        histred[justRed[i,j,0]] += 1
        histgreen[justGreen[i,j,1]] += 1
        histblue[justBlue[i,j,2]] += 1
        pixel_value = int(AG[i, j] * 255)  # Convert to 0-255 scale
        pixel_value = min(max(pixel_value, 0), 255)  # Ensure it's within 0-255 range
        histgray[pixel_value] += 1
        


import matplotlib.pyplot as plt

#plot the histogram
print("histogram red ")
plt.plot(histred, color='red')
plt.title('Histogram of red-level Image AG')
plt.plot(histred)
plt.show()

print("histogram green ")
plt.plot(histgreen, color='green')
plt.title('Histogram of green-level Image AG')
plt.plot(histgreen)
plt.show()


print("histogram blue ")
plt.plot(histblue, color='blue')
plt.title('Histogram of blue-level Image AG')
plt.plot(histblue)
plt.show()


histAG = np.histogram(AG.flatten(), bins=256, range=(0, 256))[0]
plt.plot(histAG, color='black')
plt.title('Histogram of Gray-level Image AG')
plt.show()


#part 5 binarinzing the image
TB = int(input("Enter the threshold brightness (0-255) for binarization: "))

binary_image = (AG > TB) * 255
# Display binary image
print("test")
Image.fromarray(np.uint8(binary_image)).save('binaryimage.jpg')
img = Image.open('binaryimage.jpg')
img.show()

#part 6 edge detection
TE = int(input("threshold value: "))
width, height = AG.shape
Gx = np.zeros_like(AG)
Gy = np.zeros_like(AG)
AE = np.zeros_like(AG)
for m in range (width):
    for n in range(height - 1):
       Gx[m,n] = AG[m,n+1] - AG[m,n] 

for m in range (width - 1):
    for n in range(height):
       Gy[m,n] = AG[m+1,n] - AG[m,n] 

GM = np.sqrt(Gx**2 + Gy**2)

for m in range(width - 1):
    for n in range(height):
        if GM[m,n] > TE:
            AE[m,n] = 255
        else:
            AE[m,n] = 0

plt.imshow(AE, cmap='gray')
plt.title("Edge Image AE")
plt.show()


from PIL import Image
#part 7 image pyramid
ImageAG = Image.open('gray_image.jpg')

pyramid_levels = [ImageAG]


for i in range(3):
    previous_level = pyramid_levels[-1]
    width, height = previous_level.size
    new_size = (width // 2, height // 2)

    # Create a new image for the current pyramid level
    new_level = Image.new("L", new_size) 

    # Calculate the average brightness for 2x2 blocks and update the pixels
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            block = [
                previous_level.getpixel((x, y)),
                previous_level.getpixel((x + 1, y)),
                previous_level.getpixel((x, y + 1)),
                previous_level.getpixel((x + 1, y + 1))
            ]
            average_pixel = sum(block) // 4
            new_level.putpixel((x // 2, y // 2), average_pixel)

    pyramid_levels.append(new_level)

# Save and display the resulting pyramid levels
pyramid_levels[1].save("AG2.jpg")
pyramid_levels[2].save("AG4.jpg")
pyramid_levels[3].save("AG8.jpg")

pyramid_levels[1].show()
pyramid_levels[2].show()
pyramid_levels[3].show()

