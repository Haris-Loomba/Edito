from PIL import Image
from PIL import ImageEnhance as ie
import cv2
import time
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox as ms
import numpy as np

bg = 'purple'

root = tk.Tk()
root.geometry('1200x600')
root.config(bg=bg)
root.title('Edito - Edit Videos')

class Main:
	def __init__(self):
		"""
		Triggered while creation fo object.
		Commonly used to declare Class Variables.
		"""
		self.cameraNum = tk.Entry(width=24, font=('bold', 18))
		self.resizeShape = tk.Entry(width=24, font=('bold', 18))
		self.img = np.ones((27,27,3), dtype=float)

		self.brightness = tk.Entry(width=12, font=('bold', 18))
		self.contrast = tk.Entry(width=12, font=('bold', 18))
		self.sharpness = tk.Entry(width=12, font=('bold', 18))
		self.color = tk.Entry(width=12, font=('bold', 18))
		self.thresholdBAW = 118
		self.mpDraw = mp.solutions.drawing_utils

	def saveImg(self):
		"""
		Save the Image.
		"""
		try:
			cv2.imwrite('image(edited).jpg', self.img)
			ms.showinfo('SUCCESS', 'Image saved successfully!')
			return True
		except:
			ms.showinfo('ERROR!', 'Image Couldnt be saves :(')
			return False


	def baw(self):
		"""
		Coverting Image to Black and White.
		"""
		try:
			cv2.destroyAllWindows()
			self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
			_,self.img = cv2.threshold(self.img, self.thresholdBAW, 255, cv2.THRESH_BINARY)
			cv2.imshow('Edited Image', self.img)
		except:
			ms.showinfo('Error Occured!', 'Error! Please Try again.')
			return False

	def bawInv(self):
		"""
		Coverting Image to Black and White. (Inverse)
		"""
		try:
			cv2.destroyAllWindows()
			self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
			_,self.img = cv2.threshold(self.img, self.thresholdBAW, 255, cv2.THRESH_BINARY_INV)
			cv2.imshow('Edited Image', self.img)
		except:
			ms.showinfo('Error Occured!', 'Error! Please Try again.')
			return False

	def bawTrunc(self):
		"""
		Black and White (Trunc)
		"""
		try:
			cv2.destroyAllWindows()
			self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
			_,self.img = cv2.threshold(self.img, self.thresholdBAW, 255, cv2.THRESH_TRUNC)
			cv2.imshow('Edited Image', self.img)
		except:
			ms.showinfo('Error Occured!', 'Error! Please Try Again.')
			return False

	def enhanceChanges(self):
			"""
			Function for PIL operations.
			"""
			cv2.destroyAllWindows()
			try:
				self.brightnessScale = int(self.brightness.get())
				self.contrastScale = int(self.contrast.get())
				self.sharpnessScale = int(self.sharpness.get())
				self.colorScale = int(self.color.get())
			except:
				ms.showinfo('ERROR', 'INVALID INPUTS! (Tip: Enter 1 for Default)')
				return False
			self.imPil = Image.fromarray(self.img)
			self.imPil = ie.Brightness(self.imPil).enhance(self.brightnessScale)
			self.imPil = ie.Contrast(self.imPil).enhance(self.contrastScale)
			self.imPil = ie.Sharpness(self.imPil).enhance(self.sharpnessScale)
			self.imPil = ie.Color(self.imPil).enhance(self.colorScale)
			self.img = np.array(self.imPil)

			cv2.imshow('Edited Image', self.img)

	def resizeCmd(self):
		"""
		Function for resizing
		"""
		cv2.destroyAllWindows()
		if 'x' in self.resizeShape.get():
			resized = []
			self.resizeShapeA = self.resizeShape.get().split('x')
			if len(self.resizeShapeA) == 2:
				for i in self.resizeShapeA:
					try:
						resized.append(int(i))
					except:
						ms.showinfo('ERROR', 'INVALID INPUTS!')
						return False
			else:
				ms.showinfo('ERROR', 'INVALID INPUTS!')
				return False
		else:
			ms.showinfo('ERROR', 'INVALID INPUTS!')
			return False
		self.img = cv2.resize(self.img, tuple(resized), cv2.INTER_CUBIC)
		cv2.imshow('Edited Image', self.img)

	def shoeTracking(self):
		"""
		Use Mediapipe to track shoes
		"""
		cv2.destroyAllWindows()

		mpObj = mp.solutions.objectron
		objectron = mpObj.Objectron(
			static_image_mode=False,
			max_num_objects=2,
			model_name='Shoe'
		)

		self.img.flags.writeable = False
		results = objectron.process(self.img)

		self.img.flags.writeable = True

		if results.detected_objects:
			for detected_objects in results.detected_objects:
				self.mpDraw.draw_landmarks(
					self.img,
					detected_objects.landmarks_2d,
					mpObj.BOX_CONNECTIONS,
					)

		cv2.imshow('Edited Image', self.img)

	def faceMesh(self):
		"""
		Use mediapipe for FaceMesh Algorithim.
		"""
		cv2.destroyAllWindows()
		mpFaceMesh = mp.solutions.face_mesh
		faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence = 0.4, min_tracking_confidence=0.5)
		results = faceMesh.process(self.img)

		if results.multi_face_landmarks:
			for faceLms in results.multi_face_landmarks:
				for idF, lm in enumerate(faceLms.landmark):
					h,w,c=self.img.shape
					cx, cy = int(lm.x*w), int(lm.y*h)
					if idF == 0:
						cv2.circle(self.img, (cx, cy),2,(255,0,255),-1)
				self.mpDraw.draw_landmarks(self.img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, self.mpDraw.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=1), self.mpDraw.DrawingSpec(color=(0,255,0), thickness=1,circle_radius=1))

		cv2.imshow('Edited Image', self.img)

	def poseTrack(self):
		"""
		Use Mediapipe algorithim to Pose Track.
		"""
		cv2.destroyAllWindows()
		mpPose = mp.solutions.pose
		pose = mpPose.Pose(smooth_landmarks=True, static_image_mode=False, min_detection_confidence = 0.4, min_tracking_confidence=0.4)
		results = pose.process(self.img)

		if results.pose_landmarks:
			self.mpDraw.draw_landmarks(self.img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1), self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1))

		cv2.imshow('Edited Image', self.img)

	def handTrack(self):
		"""
		Mediapipe, Hand Tracking Algorithim
		"""
		cv2.destroyAllWindows()
		mpHands = mp.solutions.hands
		hands = mpHands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence = 0.4, min_tracking_confidence=0.4)

		results = hands.process(self.img)
		if results.multi_hand_landmarks:
			for handLms in results.multi_hand_landmarks:
					for idH, lm in enumerate(handLms.landmark):
						# print(idH, lm) 
						"""
						idH refers to the id of the marker, https://google.github.io/mediapipe/solutions/hands.html
						lm reprsenting coordinates
						"""
						h,w,c=self.img.shape
						cx, cy = int(lm.x*w), int(lm.y*h)
						if idH==0:
							cv2.circle(self.img,(cx, cy),10,(255,0, 255),-1) # Drawing a circle at wrist
							
					self.mpDraw.draw_landmarks(self.img, handLms, mpHands.HAND_CONNECTIONS,self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2,circle_radius=1), self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2,circle_radius=1))

		cv2.imshow('Edited Image', self.img)

	def cupTrack(self):
		"""
		Mediapipe, Objectron (Cup Algorithim)
		"""
		cv2.destroyAllWindows()

		mpObj = mp.solutions.objectron
		objectron = mpObj.Objectron(
			static_image_mode=False,
			max_num_objects=2,
			model_name='Cup'
		)

		self.img.flags.writeable = False
		results = objectron.process(self.img)

		self.img.flags.writeable = True

		if results.detected_objects:
			for detected_objects in results.detected_objects:
				self.mpDraw.draw_landmarks(
					self.img,
					detected_objects.landmarks_2d,
					mpObj.BOX_CONNECTIONS,
					)

		cv2.imshow('Edited Image', self.img)

	def filter3(self):
		cv2.destroyAllWindows()
		k1 = np.ones((3,3))/(3*3)
		self.img = cv2.filter2D(self.img,-1,k1)
		cv2.imshow('Edited Image', self.img)

	def filter5(self):
		cv2.destroyAllWindows()
		k2 = np.ones((5,5))/(5*5)
		self.img = cv2.filter2D(self.img,-1,k2)
		cv2.imshow('Edited Image', self.img)

	def filter9(self):
		cv2.destroyAllWindows()
		k3 = np.ones((9,9))/(9*9)
		self.img = cv2.filter2D(self.img,-1,k3)
		cv2.imshow('Edited Image', self.img)

	def filter12(self):
		cv2.destroyAllWindows()
		k4 = np.ones((12,12))/(12*12)
		self.img = cv2.filter2D(self.img,-1,k4)
		cv2.imshow('Edited Image', self.img)

	def filter15(self):
		cv2.destroyAllWindows()
		k5 = np.ones((15,15))/(15*15)
		self.img = cv2.filter2D(self.img,-1,k5)
		cv2.imshow('Edited Image', self.img)

	def activationAnddisplay(self):
		"""
		This function will display everything for the image editor.
		This function will also check for the image and display it (Work on Frontend)
		Starting The Camera.
		"""
		try:
			self.img = cv2.imread(self.cameraNum.get())
			self.img = cv2.resize(self.img, (720, 480), cv2.INTER_CUBIC)
			cv2.imshow('Image', self.img)
		except:
			ms.showinfo('ERROR', 'INVALID INPUTS!')
			return False
		resizeLabel = tk.Label(text = 'Resize WxH --> (Eg, 1200x400)', bg=bg, fg='cyan', font=('bold', 18))
		resizeBtn = tk.Button(text='Resize (Current -- 720x480)', bg='yellow', fg='purple', font=(14), command=self.resizeCmd)

		resizeLabel.place(x=76, y=118)
		resizeBtn.place(x = 850, y=113)
		self.resizeShape.place(x=500,y=118)

		bawBtn = tk.Button(text='Black&White', bg='yellow', fg='purple', font=('bold', 14), command=self.baw)
		bawBtn.config(width=20)
		bayInvBtn = tk.Button(text='Black&White(Inverse)', bg='yellow', fg='purple', font=('bold', 14), command=self.bawInv)
		bayInvBtn.config(width=20)
		bayGrayBtn = tk.Button(text='Black&White(Trunc)', bg='yellow', fg='purple', font=('bold', 14), command=self.bawTrunc)
		bayGrayBtn.config(width=20)

		bawBtn.place(x=76, y=185)
		bayInvBtn.place(x = 850, y=185)
		bayGrayBtn.place(x=500,y=185)

		shoeBtn = tk.Button(text='Shoe-Tracking', bg='yellow', fg='purple', font=('bold', 14), command=self.shoeTracking)
		faceMesh = tk.Button(text='Face Mesh (Face Track)', bg='yellow', fg='purple', font=('bold', 14), command=self.faceMesh)
		fullBody = tk.Button(text='Pose Tracking (Full-Body)', bg='yellow', fg='purple', font=('bold', 14), command=self.poseTrack)
		handTrack = tk.Button(text='Hand-Tracking', bg='yellow', fg='purple', font=('bold', 14), command = self.handTrack)
		cupBtn = tk.Button(text='Cup-Tracking', bg='yellow', fg='purple', font=('bold', 14), command = self.cupTrack)

		shoeBtn.place(x=8, y=258)
		faceMesh.place(x=252, y=258)
		fullBody.place(x=497, y=258)
		handTrack.place(x=747, y=258)
		cupBtn.place(x=986, y=258)

		filterL1 = tk.Button(text='Filter(Level 1)', bg='yellow', fg='purple', font=('bold', 14), command=self.filter3)
		filterL2 = tk.Button(text='Filter(Level 2)', bg='yellow', fg='purple', font=('bold', 14), command=self.filter5)
		filterL3 = tk.Button(text='Filter(Level 3)', bg='yellow', fg='purple', font=('bold', 14), command=self.filter9)
		filterL4 = tk.Button(text='Filter(Level 4)', bg='yellow', fg='purple', font=('bold', 14), command=self.filter12)
		filterL5 = tk.Button(text='Filter(Level 5)', bg='yellow', fg='purple', font=('bold', 14), command=self.filter15)

		filterL1.place(x=8, y=326)
		filterL2.place(x=252, y=326)
		filterL3.place(x=497, y=326)
		filterL4.place(x=747, y=326)
		filterL5.place(x=986, y=326)

		brightL = tk.Label(text='Brightness (0-Infinity)', bg=bg, fg='yellow', font=(14))
		sharpL = tk.Label(text='Sharpness (0-Infinity)', bg=bg, fg='yellow', font=(14))
		contrastL = tk.Label(text='Contrast (0-Infinity)', bg=bg, fg='yellow', font=(14))
		colorL = tk.Label(text='Coloring (0-Infinity)', bg=bg, fg='yellow', font=(14))

		brightL.place(x = 0, y=440)
		sharpL.place(x = 215, y=440)
		contrastL.place(x = 429, y=440)
		colorL.place(x = 669, y=440)

		mainChangeBtn = tk.Button(text='Make Changes', bg='yellow', fg='purple', font=('bold', 14), command=self.enhanceChanges)

		self.brightness.place(x=10,y=523)
		self.sharpness.place(x=215,y=523)
		self.contrast.place(x=429,y=523)
		self.color.place(x=669,y=523)

		mainChangeBtn.config(width=15, height=1)
		mainChangeBtn.place(x=948,y=515)

		saveBtn = tk.Button(text='Save Image', bg='cyan', fg='black', font=('bold', 14), command=self.saveImg)
		saveBtn.place(x=940, y = 435)
		saveBtn.config(width=15)

	def startup(self):
		"""
		This function will display text for Selecting Images.
		"""
		header = tk.Label(text='Edito - Edit Photo', bg=bg, fg='cyan', font=('bold', 18))
		imgLabel = tk.Label(text='Enter Image Path', bg=bg, fg='yellow', font=(14))
		imgBtn = tk.Button(text='Choose Photo', bg='yellow', fg='purple', font=(14), command=self.activationAnddisplay)

		# Placement
		header.pack()
		imgLabel.place(x=80, y=53)
		imgBtn.place(x = 800, y=48)
		self.cameraNum.place(x=406,y=53)


main = Main()
main.startup()

root.mainloop()# Main Loop