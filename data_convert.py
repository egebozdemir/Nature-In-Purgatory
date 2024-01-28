# #DAILY PRACTISE#
# L1 = []
# L1.append([1, [2, 3], 4])
# L1.extend([7, 8, 9])
# print(L1[0][1][1] + L1[2])

from PIL import Image
import os

data_dir =  'raw'
#target_dir = 'formatted'
target_dir = 'dataset'
for file in os.listdir(data_dir):
    #print (file)
    f_img = data_dir+"/"+file
    #print(f_img)
    img = Image.open(f_img)
    img = img.resize((1280, 1024)) #same resolution of original .bmp images
    (name, extension) = os.path.splitext(file)
    if(file<"A632 - 20221103_020148"):
        name = name[:4] + "_dead"
    else:
        name = name[:4] + "_alive"
    #print (name)
    img.save(target_dir+"/"+name+".png", 'png')



