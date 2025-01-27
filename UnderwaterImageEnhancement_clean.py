# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import glob
from tkinter import *
from tkinter import ttk 
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import datetime
import random
import math
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
from skimage.io import imshow 
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report
from skimage import exposure
plt.gray()



def mainFrame():
	# =====> HomeFrame destroy
	homeFrame.destroy()
	# =====> HomeFrame destroy

	def showPlotImg(row, col, im, titleText=''):
		fig = Figure(figsize = (2, 2),dpi = 100)
		plot1 = fig.add_subplot()
		plot1.axes.get_xaxis().set_visible(False)
		plot1.axes.get_yaxis().set_visible(False)
		plot1.imshow(im)
		plot1.set_title(titleText,fontsize=10)
		canvas = FigureCanvasTkAgg(fig,master = upLeftFrame)
		canvas.draw()
		canvas.get_tk_widget().grid(row=row, column=col,padx=5, pady=8)

	def placeholder():
		titles = [["Original Image", "Resized Image" , "Red Image", "Green Image","Blue Image"],
		["WB- Red Image", " WB- Green Image", " WB- Blue Image", "HE-Red Image" , "HE- Green Image" ],
		[" HE- Blue Image", "Gamma Corrected-R Band" , "Gamma Corrected G- band", "Gamma Corrected B- band", "Enhanced image"]]
		img = mpimg.imread(r'GUI_images/placeHolder.png')
		for i in [0,1,2]:
			for j in [0,1,2,3,4]:
				showPlotImg(i, j, img, titles[i][j])



	def browse_button():
		global img
		filename = askopenfilename()
		img = mpimg.imread(filename)
		showPlotImg(0, 0, img, 'ORIGINAL IMAGE')

	def preprocessing_button():
		global r
		global g
		global b
		global resized_image
		h1=300
		w1=300

		dimension = (w1, h1) 
		resized_image = cv2.resize(img,(h1,w1))
		showPlotImg(0, 1, resized_image, 'Resized Image')

		r = resized_image[:,:,0] *255
		g = resized_image[:,:,1] *255
		b = resized_image[:,:,2] *255

		r = r.astype('uint8')
		g = g.astype('uint8')
		b = b.astype('uint8')

		showPlotImg(0, 2, r, 'Red Image')
		showPlotImg(0, 3, g, 'Green Image')
		showPlotImg(0, 4, b, 'Blue Image')


	def whitebalance_button():
		global rw
		global gw
		global bw
		r_avg = cv2.mean(r)[0]
		g_avg = cv2.mean(g)[0]
		b_avg = cv2.mean(b)[0]

		k = (r_avg + g_avg + b_avg) / 3
		kr = k / r_avg
		kg = k / g_avg
		kb = k / b_avg
		
		rw = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
		gw = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
		bw = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

		showPlotImg(1, 0, rw, 'WB-Red Image')
		showPlotImg(1, 1, gw, 'WB-Green Image')
		showPlotImg(1, 2, bw, 'WB-Blue Image')


	def histogramEqualization_button():
		global equ1
		global equ2
		global equ3
		equ1 = cv2.equalizeHist(rw,0.01)
		showPlotImg(1, 3, equ1, 'HE-Red Image')

		equ2 = cv2.equalizeHist(gw,0.01)
		showPlotImg(1, 4, equ2, 'HE-Green Image')

		equ3 = cv2.equalizeHist(bw,0.01)
		showPlotImg(2, 0, equ3, 'HE-Blue Image')


	def GammaCorrection_button():
		global gamma_corrected_red
		global gamma_corrected_green
		global gamma_corrected_blue
		global SEG_im

		SEG_im = np.zeros((300,300,3)).astype(int)
		img_red = equ1
		gamma_corrected_red = exposure.adjust_gamma(img_red, 0.5)  
		showPlotImg(2, 1, gamma_corrected_red, 'Gamma Corrected- R Band')

		img_green = equ2
		gamma_corrected_green = exposure.adjust_gamma(img_green, 0.5)  
		showPlotImg(2, 2, gamma_corrected_green, 'Gamma Corrected G- band')

		img_blue = equ3
		gamma_corrected_blue = exposure.adjust_gamma(img_blue, 0.5)
		showPlotImg(2, 3, gamma_corrected_blue, 'Gamma Corrected B- band')


	def CNN_button():
		train = glob.glob(r'Dataset\train\*.png')
		test=glob.glob(r'Dataset\test\*.png')
		val = glob.glob(r'Dataset\val\*.png')

		lst_train = []
		for x in train:
			lst_train.append([x,0])

		lst_test = []
		for x in test:
			lst_test.append([x,1])

		lst_val = []
		for x in val:
			lst_val.append([x,2])
		lst_complete = lst_train + lst_val
		random.shuffle(lst_complete)

		df = pd.DataFrame(lst_complete,columns = ['files','target'])
		df.head(10)
		filepath_img ="train/test/val/*.png"
		df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
		df.shape

		def preprocessing_image(filepath):
			img = cv2.imread(filepath) #read
			img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
			img = cv2.resize(img,(56,56))  # resize
			img = img / 255 #scale
			return img

		def create_format_dataset(dataframe):
			X = []
			y = []
			for f,t in dataframe.values:
				X.append(preprocessing_image(f))
				y.append(t)
			
			return np.array(X),np.array(y)
		X, y = create_format_dataset(df)
		X.shape,y.shape
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y)
		X_train.shape,X_test.shape,y_train.shape,y_test.shape


		'''CNN'''
		CNN = Sequential()

		CNN.add(Conv2D(32,(2,2),input_shape = (56,56,3),activation='relu'))
		CNN.add(Conv2D(64,(2,2),activation='relu'))
		CNN.add(MaxPooling2D())
		CNN.add(Conv2D(32,(2,2),activation='relu'))
		CNN.add(MaxPooling2D())

		CNN.add(Flatten())
		CNN.add(Dense(32))
		CNN.add(Dense(1,activation= "sigmoid"))
		CNN.summary()
		CNN.compile(optimizer='adam',loss = 'mse',metrics=['accuracy'])
		CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 5,batch_size = 30).history
		history = CNN.history.history
		mseLossLabel.config(text='Mean Square Error loss is : {}'.format(history['loss'][4]))
		cnnAccLabel.config(text="Accuracy of the CNN is : {} %".format(CNN.evaluate(X_test,y_test)[1]*100))
		train_loss = history['loss']
		val_loss = history['val_loss']
		train_acc = history['accuracy']
		val_acc = history['val_accuracy']


	def analysis_button():
		SEG_im[:,:,0] = gamma_corrected_red
		SEG_im[:,:,1] = gamma_corrected_green
		SEG_im[:,:,2] = gamma_corrected_blue

		showPlotImg(2, 4, SEG_im, 'Enhanced image')

		summed = np.mean((resized_image - SEG_im) ** 2)
		num_pix=resized_image.shape[0]*SEG_im.shape[1]
		mse=summed/num_pix

		max_pixel = 255.0
		mseLabel.config(text='MSE : {}'.format(mse))

		from math import log10, sqrt

		psnr = 20 * log10(max_pixel / sqrt(mse))
		psnrLabel.config(text='PSNR : {}'.format(psnr))


	# ========= GUI part  ========= #

	leftFrame = Frame(root, height=700, width=1050,bg = 'grey')
	leftFrame.grid(row=0,column=0)
	leftFrame.grid_propagate(0)

	rightFrame = Frame(root, height=700, width=200,bg = 'white')
	rightFrame.grid(row=0,column=1)
	rightFrame.grid_propagate(0)


	# =====> Left main frame
	upLeftFrame = Frame(leftFrame, height=650, width=1050)
	upLeftFrame.grid(row=0,column=0)
	upLeftFrame.grid_propagate(0)

	downLeftFrame = Frame(leftFrame, height=100, width=1050)
	downLeftFrame.grid(row=1,column=0)
	downLeftFrame.grid_propagate(0)
	# =====> Left main frame

	# =====> Butoons  upLeftFrame
	InputimageBtn = Button(rightFrame,text = "Input Image",command=browse_button, bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	InputimageBtn.grid(row=0, column=0,sticky='w',padx=20, pady=30)

	PreprocessingBtn = Button(rightFrame,text = "Preprocessing", command=preprocessing_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	PreprocessingBtn.grid(row=1, column=0,sticky='w',padx=20, pady=30)

	whiteBalanceBtn = Button(rightFrame,text = "White Balance", command=whitebalance_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	whiteBalanceBtn.grid(row=2, column=0,sticky='w',padx=20, pady=30)

	histigramEqualizationBtn = Button(rightFrame,text = "Histigram Equalization",command=histogramEqualization_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	histigramEqualizationBtn.grid(row=3, column=0,sticky='w',padx=20, pady=30)

	gammaCorrectionBtn = Button(rightFrame,text = "Gamma Correction",command=GammaCorrection_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	gammaCorrectionBtn.grid(row=4, column=0,sticky='w',padx=20, pady=30)

	CnnBtn = Button(rightFrame,text = "CNN",command=CNN_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	CnnBtn.grid(row=5, column=0,sticky='w',padx=20, pady=30)

	analysisBtn = Button(rightFrame,text = "analysis",command=analysis_button,  bg='brown',fg='white', width = 20, font = ("Helvetica",10,"bold") )
	analysisBtn.grid(row=6, column=0,sticky='w',padx=20, pady=30)
	# =====> Butoons  upLeftFrame

	# =====> label  downLeftFrame
	mseLossLabel = Label(downLeftFrame, text="Mean Square Error loss is : NA",font = ("Helvetica",10,"bold"))  
	mseLossLabel.grid(row=0,column=0, padx=30)
	cnnAccLabel = Label(downLeftFrame, text="Accuracy of the CNN is : NA",font = ("Helvetica",10,"bold"))  
	cnnAccLabel.grid(row=1,column=0, padx=30)

	mseLabel = Label(downLeftFrame, text="MSE : NA",font = ("Helvetica",10,"bold")) 
	mseLabel.grid(row=0,column=1, padx=30)
	psnrLabel = Label(downLeftFrame, text="PSNR : NA",font = ("Helvetica",10,"bold"))  
	psnrLabel.grid(row=1,column=1, padx=30)
	# =====> label  downLeftFrame
	placeholder()

root = Toplevel()
root.geometry("1250x700")
root.title("Underwater Image Enhancement")

homeFrame = Frame(root, height='700', width='1250', bg='blue')
homeFrame.grid(row=0, column=0)
homeFrame.grid_propagate(0)

img = Image.open(r"GUI_images/home.png")
img = ImageTk.PhotoImage(img)
panel = Label(homeFrame, image=img)
panel.pack(side="top", fill="both", expand="yes")


startImg = Image.open(r"GUI_images/startbutton.png")
startImg = ImageTk.PhotoImage(startImg)
browsebtn = Button(homeFrame,text = "" ,image = startImg,bg='white',relief = FLAT, borderwidth='0', command=mainFrame)
browsebtn.place(x=490,y = 570)

root.mainloop()