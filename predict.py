import sys
import cv2
from keras.models import load_model
import numpy as np
from datasets import get_labels
from preprocess import preprocess_input
from keras.preprocessing import image
import math
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Label
from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
#from tkFileDialog import askopenfilename



#parameters for loading data and images

root = tk.Tk()
root.title("STRESS EVALUATION MODEL")
root.geometry("650x650")
root.configure(background='white')
back = tk.Frame(master=root,bg='black')

#swin = ScrolledWindow(root, width=500, height=500)
#swin.pack()
#win = swin.window

def clicked():
		
	label = tk.Label(root)
	#load the file path
	image_path = filedialog.askopenfilename()
	#print(image_path)
	#open the image
	image_path1 = Image.open(image_path)
	image_path1 = image_path1.resize((200, 200), Image.ANTIALIAS)
	label.image_path1 = ImageTk.PhotoImage(image_path1)
	label['image'] = label.image_path1
	label.pack()

	detection_model_path = '/Users/Ramanuj/Downloads/stress_detection-master/trained_models/detection_models/haarcascade_frontalface_default.xml'
	emotion_model_path = '/Users/Ramanuj/Downloads/stress_detection-master/trained_models/emotion_modelsfer2013_mini_XCEPTION.86-0.43.hdf5'
	emotion_labels = get_labels('fer2013')

	#loading models
	face_detection = cv2.CascadeClassifier(detection_model_path)
	emotion_classifier = load_model(emotion_model_path, compile=False)

	# getting input model shapes for inference
	emotion_target_size = emotion_classifier.input_shape[1:3]



	# loading images
	pil_image = image.load_img(image_path, grayscale=True)
	gray_image = image.img_to_array(pil_image)
	gray_image = np.squeeze(gray_image)
	gray_image = gray_image.astype('uint8')

	faces = face_detection.detectMultiScale(gray_image, 1.3, 5)
	if (len(faces)==0):
		print("Please enter a valid image!")
		z = ttk.Label(root, text='Please enter a valid image!',background="white")
		z.pack()
	
	var = 0
	for (x, y, w, h) in faces:
		gray_face = gray_image[y:y+w , x:x+h]
		gray_face = cv2.resize(gray_face, (emotion_target_size))
		gray_face = preprocess_input(gray_face)
		gray_face = np.expand_dims(gray_face, 0)
		gray_face = np.expand_dims(gray_face, -1)
		emotion_proba = emotion_classifier.predict(gray_face)
		print("------------------------------------------------------------------------")
		print("Probabilities of each class: ")
		print(" 0:angry , 1:disgust , 2:fear , 3:happy , 4:sad , 5:surprise , 6:neutral ")
		print(emotion_proba)
		emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
		emotion_text = emotion_labels[emotion_label_arg]
		print("-------------------------------------------------------------------------")
		print("Emotion class: ")	
		print(emotion_text)
		print("--------------------------------------------------------------------------")
		var = np.amax(emotion_proba)
		print("Maximum probability emotion: ")
		print(var)
		c = ttk.Label(root, text='MAXIMUM PROBABILITY EMOTION: ',background="white")
		c.pack()
		d = ttk.Label(root, text=var,background="white")
		d.pack()
		f = ttk.Label(root, text="EMOTION LABEL: ",background="white")
		f.pack()
		e = ttk.Label(root, text=emotion_text,background="white")
		e.pack()
		
		
		var = var * 100
		if(emotion_text == "angry"):
			S = 2.36*(math.log(0.33*var + 1.00))

		elif(emotion_text == "disguist"):
			S = 7.27*(math.log(0.01*var + 1.02))

		elif(emotion_text == "fear"):
			S = 1.76*(math.log(1.36*var + 1.00))

		elif(emotion_text == "happy"):
			S = -7.56*(math.log(-0.003*var + 1.01))

		elif(emotion_text == "sad"):
			S = 2.85*(math.log(0.13*var + 1.01))

		elif(emotion_text == "surprise"):
			S = 2.45*(math.log(0.29*var + 1.00))
			
		elif(emotion_text == "neutral"):
			S = 5.05*(math.log(0.015*var + 1.016))

		print("Stress Value:")
		print(S)
		a = ttk.Label(root, text='STRESS VALUE(range:0-9): ',background="white")
		a.pack()
		b = ttk.Label(root, text=S,background="white")
		b.pack()
	
w = Label(root, text="STRESS DETECTION MODEL", font=("Times new roman", 16),fg="blue", background="white")
w.pack()
b1 = tk.Button(root, text = "Upload Image", command = clicked, compound=LEFT,background="white")
b1.pack()
b2 = tk.Button(root, text = "Exit", command = lambda : root.quit(), compound=LEFT,background="white")
b2.pack()
root.mainloop()
