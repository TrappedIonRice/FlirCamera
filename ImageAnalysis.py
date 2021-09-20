import matplotlib.pyplot as plt
import PIL
import numpy as np
from PIL import Image

image1 = Image.open('parallel_lines_target.jpg')
data1 = np.asarray(image1)
cropped1 = data1[300:2200,1000:3000]
#print(np.shape(cropped1))

#image2 = Image.fromarray(cropped)
xvals1 = list(range(0,np.shape(cropped1)[1]))
means1 = cropped1.mean(axis=0)

#plt.plot(xvals1, means1)
#plt.show()

image2 = Image.open('parallel_horizontal.jpg')
data2 = np.asarray(image2)
cropped2 = data2[350:2200,1000:3000]

Image.fromarray(cropped2).show()


xvals2 = list(range(0,np.shape(cropped2)[0]))
means2 = cropped2.mean(axis=1)

plt.plot(xvals2, means2)
plt.show()


