import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np

from PIL import Image, ImageTk
from keras.models import load_model
from tkinter import Tk, Button, filedialog,Label
import tensorflow as tf



dic = {0:'cat', 1 :'dog'}
img_size=256

root=tk.Tk()
root.title("disting cats and dogs")
root.geometry('600x600')
root.configure(bg="red")

imgFrame=tk.Frame(height=480,width=480) #create the container where the vedio will be displayed
imgFrame.pack()
img=ctk.CTkLabel(imgFrame)
img.pack()

imgtk=None


model=load_model('cat_dog.h5')


def select_image():
    global image_path,imgtk
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    image_path = file_path
    image = Image.open(file_path)
    imgtk = ImageTk.PhotoImage(image)
    img.imgtk=imgtk  #associate the imgtk image with the vid object.
    img.configure(image=imgtk)
    
   



def predict_label(image_path): 
    img=cv2.imread(image_path)
    resized=cv2.resize(img,(img_size,img_size)) 
    normalized = tf.keras.utils.img_to_array(resized)/255.0
    normalized = normalized.reshape(1,img_size,img_size,3)
    pred = model.predict(normalized)
    return dic[np.argmax(pred)]
prediction_label = None
def predict():
    global image_path, prediction_label
    
    prediction_text = predict_label(image_path)
        
    if prediction_label is None:
        # Créez un label pour afficher le texte de prédiction
        prediction_label = Label(root, text=prediction_text)
        prediction_label.pack()
    else:
        # Mettez à jour le contenu de l'étiquette de prédiction
        prediction_label.config(text=prediction_text)

select_button = Button(root, text="Sélectionner une image", command=select_image)
select_button.pack()
predict_button = Button(root, text="Predict", command=predict)
predict_button.pack()





root.mainloop()