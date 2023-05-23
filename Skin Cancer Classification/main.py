#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from keras.utils.np_utils import to_categorical 
from keras.models import load_model
from sklearn.model_selection import train_test_split


# ## 2-Read Pickle File

# In[ ]:


df = pd.read_pickle("df.pkl") # verilerimizi hızlıca yukarıdaki adımları yapmaya gerek kalmadan bu bloku debug ederiz, dataFrame nesnesine dönüştürürek.


# In[ ]:


x_train = np.asarray(df["image"].tolist())
#  stardardization
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean)/x_train_std


# # 4- Building the Model => Bir Deep Learning Algoritması: CNN

# In[ ]:


# load models
model1 = load_model("my_model1.h5")
model2 = load_model("my_model2.h5")


# # Skin Cancer Classification GUI

# In[ ]:


## global variables
img_name = ""
count = 0
img_jpg = ""


# In[ ]:


# parent widget => window
window = tk.Tk()
window.geometry("1080x640")
window.wm_title("Skin Cancer Classification")

## frames
frame_left = tk.Frame(window, width = 540, height = 640, bd = "2") #border size=bd=sınırlarının genişliği
frame_left.grid(row = 0, column = 0)

frame_right = tk.Frame(window, width = 540, height = 640, bd = "2")
frame_right.grid(row = 0, column = 1)

frame1 = tk.LabelFrame(frame_left, text = "Image", width = 540, height = 500)
frame1.grid(row = 0, column = 0)

frame2 = tk.LabelFrame(frame_left, text = "Model and Save", width = 540, height = 140)
frame2.grid(row = 1, column = 0)

frame3 = tk.LabelFrame(frame_right, text = "Features", width = 270, height = 640)
frame3.grid(row = 0, column = 0)

frame4 = tk.LabelFrame(frame_right, text = "Result", width = 270, height = 640)
frame4.grid(row = 0, column = 1, padx = 10)


# In[ ]:


# frame1
def imageResize(img):
    # For example: img.size= 1000x1200
    basewidth = 500

    # img.size[0] = 1000, So, 500/1000 = 0.5 = wpercent
    wpercent = (basewidth/float(img.size[0]))    

    # img.size[1] = 1200, So, 1200*0.5 = 600 = hsize
    hsize = int((float(img.size[1])*float(wpercent))) 

    # img.resize boyutları => 500x600 olmuş oldu
    img = img.resize((basewidth, hsize),Image.ANTIALIAS) # resize yaparken araları doldurmak için antialiasing yöntemi kullanılır. Ayrıntıya gerek yok.
    return img


# In[ ]:


def openImage():
    
    global img_name
    global count # yalnızca bir resim görüntülenmesi sağlamak için
    global img_jpg
    
    count += 1
    if count != 1:
        messagebox.showinfo(title = "Warning", message = "Only one image can be opened")
    else:
        img_name = filedialog.askopenfilename(initialdir = "img",title = "Select an image file")
        
        #02-Python GUI - Tkinter and PyQt5 with Real World Python Projects/04-Skin Cancer Classification Project with Tkinter/img/ISIC_0024306.jpg
        img_jpg = img_name.split("/")[-1].split(".")[0] # ISIC_0024306 (resim adı alınır)
        # image label
        tk.Label(frame1, text =img_jpg, bd = 3 ).pack(pady = 10)
    
        # open and show image
        img = Image.open(img_name)
        img = imageResize(img)
        img = ImageTk.PhotoImage(img) # Tk kütüphanesiyle kullanabilmek için uygun formata sokmak zorundayız. 
        panel = tk.Label(frame1, image = img) # frame1'de duracak olan resmi belirttik.
        panel.image = img
        panel.pack(padx = 15, pady = 10)
        
        # image feature
        data = pd.read_csv("data\HAM10000_metadata.csv")
        cancer = data[data.image_id == img_jpg]
        # print(cancer) dersek aşağıdaki gibi output verir. Size'ı 7 dir.
        # lesion_id   image_id      dx  dx_type     age  sex   localization
        # HAM_0000550 ISIC_0024306  nv  follow_up  45.0  male    trunk

        for i in range(cancer.size):
            x = 0.4 # labellar 0.1di biraz sağa koyuyorum
            y = (i/10)/2
            tk.Label(frame3, font = ("Times",12), text = str(cancer.iloc[0,i])).place(relx = x, rely = y)


# In[ ]:


# menu
menubar = tk.Menu(window)
window.config(menu = menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label = "File",menu = file)
file.add_command(label = "Open", command = openImage)


# In[ ]:


# frame3
def classification():
    
    if img_name != "" and models.get() != "": #image ve modeller boş olmaması lazım
        
        # model selection
        if models.get() == "Model1":
            classification_model = model1 # cnn'de kaydediğimiz model1
        else:
            classification_model = model2 # cnn'de kaydediğimiz model1
        
        z = df[df.image_id == img_jpg] # filtering
        z = z.image.values[0].reshape(1,75,100,3) #values[0]=image_id column
        
        # datasetini train etmeden önce standardize etmiştik, aynı şekilde predict edilecek datayı da standardize ederiz.
        z = (z - x_train_mean)/x_train_std

        pred = classification_model.predict(z)[0]
        p_index = np.argmax(pred)
        p_filter = df["dx_idx"]==p_index
        pred_cancer=df["dx"][p_filter].unique()
        pred_cancer[0]
    
        for i in range(len(pred)):
            x = 0.5
            y = (i/10)/2
            
            if i != p_index: #outputu farklı renkte yapacağız.
                tk.Label(frame4,text = str(pred[i])).place(relx = x, rely = y)
            else:
                tk.Label(frame4,bg = "yellow",text = str(pred[i])).place(relx = x, rely = y)
        
        if chvar.get() == 1: #sonucları kaydedeceğiz. defaut 0 ayarlamıştık. 0 => seçili değil, 1 => seçili anlamına gelir.
            
            val = entry.get()
            entry.config(state = "disabled") # enrtry'yi get ettikten sonra, disabled yapacağız.
            path_name = val + ".txt" # result1.txt
            
            save_txt = img_name + "--" + str(pred_cancer[0])
            
            text_file = open(path_name,"w")
            text_file.write(save_txt)
            text_file.close()
        else:
            print("Save is not selected")
    else:
        messagebox.showinfo(title = "Warning", message = "Choose image and Model First!")
        tk.Label(frame3, text = "Choose image and Model First!" ).place(relx = 0.1, rely = 0.6)


# In[ ]:


columns = ["lesion_id","image_id","dx","dx_type","age","sex","localization"]
for i in range(len(columns)):
    x = 0.1
    y = (i/10)/2 #featureslerı alt alta artacak şekilde yazdırma için
    tk.Label(frame3, font = ("Times",12), text = str(columns[i]) + ": ").place(relx = x, rely = y)


# In[ ]:


classify_button = tk.Button(frame3, bg = "red", bd = 4, font = ("Times", 13),activebackground = "orange", text = "Classify", command = classification)
classify_button.place(relx = 0.1, rely = 0.5)


# In[ ]:


# frame 4 => classication labels
labels = df.dx.unique()

for i in range(len(labels)):
    x = 0.1
    y = (i/10)/2
    p_filter = df["dx_idx"]==i
    pred_cancer=df["dx"][p_filter].unique()
    tk.Label(frame4, font = ("Times",12), text = str(pred_cancer[0]) + ": ").place(relx = x, rely = y)


# In[ ]:


# frame 2 
# combo box
model_selection_label = tk.Label(frame2, text = "Choose classification model: ")
model_selection_label.grid(row = 0, column = 0, padx = 5)

models = tk.StringVar() #modelerimi string variable olarak okuyacağız.
model_selection = ttk.Combobox(frame2, textvariable = models, values = ("Model1","Model2"), state = "readonly")
model_selection.grid(row = 0, column = 1, padx = 5)

# check box
chvar = tk.IntVar()
chvar.set(0) # default 0
xbox = tk.Checkbutton(frame2, text = "Save Classification Result", variable = chvar)
xbox.grid(row = 1, column =0 , pady = 5)

# entry
entry = tk.Entry(frame2, width = 23)
entry.insert(string = "Saving name...",index = 0) #placeholder => txt file name
entry.grid(row = 1, column =1 )


# In[ ]:


window.mainloop()

