import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochs.h5')

image=cv2.imread('C:\\xampp\\htdocs\\Brain Tumor Classification\\pred\\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)
#print(img)

input_img=np.expand_dims(img, axis=0)

#result=model.predict_classes(input_img)
#print(result)

#result= np.argmax(model.predict(input_img), axis= -1)
#print(result)

predictions= model.predict(input_img)
print(predictions[0])
#predicted_label= np.argmax(predictions)
#print(predicted_label)
#labels= train_generator.class_indices
#labels= {v: k for k, v in labels.items()}
#print(labels)
