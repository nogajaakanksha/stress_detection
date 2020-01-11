# stress_detection
A Deep learning approach to detect human emotion and stress using CNN and logistic regression in python. Based on keras and pandas libraries.


####Descrption
The process is divided into two parts:
1. Facial emotion recognition
      Used Convolutional Neural Network to find the emotion category.
2. Stress detection from the deciphered emotions.
      Applied regression analysis 
      
####Important Notes:

1. You need to install tensorflow, keras library, tkinter library, cv2 and pandas.
2. The training has been done using 65 epochs. 
   You can change the number of epochs in the training_model.py file.
   To increase the accuracy, you can increase epochs and number of CNN layers and run training_model.py file.   


####Usage
1. First activate the tensorflow library using below command:
-----source ~/tensorflow/bin/activate
2. Run the predict.py file
-----python predict.py
After execuing this file, you will be prompted for input (image).
Upload the image and you will be getting the output as emotion label and stress value.
