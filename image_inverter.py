# WORKS
import cv2
import os
from PIL import Image, ImageChops

lst2=['0','1','2','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
lst=['K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
lst3=['K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']

for i in lst3:
    path=r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\data2\output\val\{}".format(i)
    os.chdir(path)
    count=0
    for count, filename in enumerate(os.listdir(path)):
        f_name=filename.format(count)
        img=Image.open(f_name)
        inv_img=ImageChops.invert(img)
        #inv_img.show()
        inv_img = inv_img.save(f_name)
            

'''cv2.imwrite("0.jpg",inv_img)
cv2.imshow("Original Image",image)
cv2.imshow("Inverted Image",inverted_image)'''
#%%

'''
os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\data2\output\train\D")
count=0
path="E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\data2\output\train\A\{}".format(count)
for i in range(0,696):
    f_name='{}.jpg'.format(count)
    img=Image.open(f_name)
    inv_img=ImageChops.invert(img)
    #inv_img.show()
    inv_img = inv_img.save(f_name)
    count+=1
'''