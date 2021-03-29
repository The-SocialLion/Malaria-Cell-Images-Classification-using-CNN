import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import ImageOps
from tensorflow.keras.preprocessing import image# used for preproccesing 
# loading or reading the saved model ,weights
model=load_model("MCI.h5")
print("loaded model from disk")
Img=64
#classification of images
def classify(img_file):
    test_image=image.load_img(img_file)
    test_image=ImageOps.grayscale(test_image)
    test_image = test_image.resize((Img,Img))
    test_image=np.expand_dims(test_image,axis=0)
    test = np.array(test_image).reshape(-1,Img,Img,1)
    result=model.predict(test)
    if result[0][0]==0:
        prediction='Parasitized'
    else:
        prediction='Uninfected'
    print("The prediction is {0},for the image {1}".format(prediction,img_file))
#storing the images in this folder
import os
path='D:/python/dl programs/CNN/Malaria Cell images/test'
files=[]
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' or '.JPG' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
