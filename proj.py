import cv2
import cv2.cv as cv
import numpy as np
import ImageDraw
import Tkinter as TK
import tkSimpleDialog
import ImageTk as iTK	
import Image as i

np.set_printoptions(threshold=np.nan)

class Gui(TK.Frame):
    
	def __init__(self,master):
	        TK.Frame.__init__(self,master)
	        self.grid()
	        self.objLst = []
	        self.currentLst = []
	        self.warpPosition = []
	        self.create_widgets()
	        self.cameraPosition = np.matrix(np.zeros([1,3]))

	def clickCallback(self,event):
		print (event.x,event.y)
		x = tkSimpleDialog.askinteger("x - Axis", "Please enter the x-axis value.")
		y = tkSimpleDialog.askinteger("y - Axis", "Please enter the y-axis value.")
		z = tkSimpleDialog.askinteger("z - Axis", "Please enter the z-axis value.")
		
		if x != None and y != None and z != None:
			self.canvas.create_rectangle(event.x-1,event.y-1,event.x+1,event.y+1)
			self.currentLst.append(np.array([event.x, event.y]))
			self.warpPosition.append(np.array([x, y, z]))
			if len(self.currentLst)>1:
				self.canvas.create_line(self.currentLst[-2][0], self.currentLst[-2][1],self.currentLst[-1][0],self.currentLst[-1][1])

	def create_widgets(self):
		im = i.open('project.jpeg')
		self.cv2Img = cv2.imread('project.jpeg')
		image = im.resize((650, 500), i.ANTIALIAS)
		self.img = iTK.PhotoImage(image)
		#self.img = self.img.subsample(2,2)
		self.canvas = TK.Canvas(self,width=650,height=500)
		self.canvas.create_image(325,250,imag=self.img)
		#self.canvas.itemconfigure(self.img,state=NORMAL)
		self.canvas.grid()
		self.canvas.bind("<Button 1>",self.clickCallback)
		self.button=TK.Button(self)
		self.button["text"]="join"
		self.button["command"]=self.joinPoints
		self.button.grid()
		self.button=TK.Button(self)
		self.button["text"]="Homography Matrix"
		self.button["command"]=self.homographyMatrix
		self.button.grid()
	
	def homographyMatrix(self):
		currentPosition = np.transpose(np.matrix(np.array(self.currentLst)))
		
		temp = np.array(self.warpPosition)
		newPoisition = np.ones([len(temp[:, 0]), len(temp[0, :]) - 1])
		for j in range(len(temp[0, :]) - 1):
			newPoisition[:, j] = temp[:, j]
		
		newPosition = np.transpose(np.matrix(newPoisition))
				
		print "newPosition = \n", newPosition
		print "currentPosition = \n", currentPosition
		
		homographyMatrix = self.findHomography(currentPosition, newPosition)
		
		print "homographyMatrix = \n", homographyMatrix
		
		# Perform transformation using the Homograhy matrix
		outImage = i.open("out.png").convert("RGBA")
		pixels = outImage.load()
		
		left = currentPosition[0, :].min()
		right = currentPosition[0, :].max()
		top = currentPosition[1, :].min()
		bottom = currentPosition[1, :].max()
		
		currentPoints = []
		currentPointsColor = []
		for y in range(top, bottom + 1):
			for x in range(left, right + 1):
				if pixels[x, y][3] != 0:
					currentPoints.append((x, y))
					currentPointsColor.append((pixels[x, y][0], pixels[x, y][1], pixels[x, y][2]))
		
		currentPoints = np.transpose(np.matrix(np.array(currentPoints)))
		currentPointsColor = np.transpose(np.matrix(np.array(currentPointsColor)))

		print "currentPoints = \n", currentPoints
		
		newPoints = self.homographyWarp(currentPoints, homographyMatrix)
		
	def homographyWarp(self, currentPoints, homographyMatrix):
		x = currentPoints[0, :]
		y = currentPoints[1, :]
		
		currentPoints = np.matrix(np.ones([3, currentPoints.shape[1]]))
		currentPoints[0] = x
		currentPoints[1] = y
		
		newPoints = np.matrix(homographyMatrix) * np.matrix(currentPoints)
		
		row1 = np.array(newPoints[0,:])
		row2 = np.array(newPoints[1,:])
		row3 = np.array(newPoints[2,:])
		
		row1 = row1 / row3
		row2 = row2 / row3
	
		newPoints = np.concatenate((row1, row2))
		print "newPoints = \n", newPoints
		
		return newPoints
	
	def findHomography(self, currentPosition, newPosition):
		if currentPosition.shape[1] != newPosition.shape[1]:
			return "Points matrices different in sizes."
		
		if newPosition.shape[0] != 2:
			return "Points matrices must have two rows."

		if newPosition.shape[1] < 4:
			return "Need at least 4 matching points." 
	
		# Solve equations using Singular Values Decomposition (SVD)
		x = newPosition[0, :]
		y = newPosition[1, :]
		x_prime = currentPosition[0, :]
		y_prime = currentPosition[1, :]
		
		h = []
		
		for j in range(0, newPosition.shape[1]):
			row1 = np.array([[currentPosition[0, j], currentPosition[1, j], 1, 0, 0, 0, -1 * newPosition[0, j] * currentPosition[0, j], -1 * newPosition[0, j] * currentPosition[1, j], -1 * newPosition[0, j]]])
			row2 = np.array([[0, 0, 0, currentPosition[0, j], currentPosition[1, j], 1, -1 * newPosition[1, j] * currentPosition[0, j], -1 * newPosition[1, j] * currentPosition[1, j], -1 * newPosition[1, j]]])
			row12 = np.concatenate([row1, row2])

			if j == 0:
				h = row12
			else:
				h = np.concatenate((h, row12))
		
		U, s, V = np.linalg.svd(np.matrix(h))
		
		homographyMatrix = np.reshape(V[8], (3, 3))
		
		return homographyMatrix
	
	def joinPoints(self):
		if(len(self.currentLst)>2):
			self.canvas.create_line(self.currentLst[-1][0], self.currentLst[-1][1],self.currentLst[0][0],self.currentLst[0][1])
			# read image as RGB and add alpha (transparency)
			im = i.open("project.jpeg").convert("RGBA")
			im = im.resize((650, 500), i.ANTIALIAS)

			# convert to numpy (for convenience)
			imArray = np.asarray(im)

			# create mask
			polygon = []
			for j in range(0, len(self.currentLst)):
				polygon.append((self.currentLst[j][0], self.currentLst[j][1]))

			maskIm = i.new('L', (imArray.shape[1], imArray.shape[0]), 0)
			ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
			mask = np.array(maskIm)

			# assemble new image (uint8: 0-255)
			newImArray = np.empty(imArray.shape,dtype='uint8')

			# colors (three first columns, RGB)
			newImArray[:,:,:3] = imArray[:,:,:3]

			# transparency (4th column)
			newImArray[:,:,3] = mask*255

			# back to Image from numpy
			newIm = i.fromarray(newImArray, "RGBA")
			newIm.save("out.png")
			
			self.objLst.append(self.currentLst)
			# self.currentLst=[]
	'''
	def joinPointsOriginal(self):
		if(len(self.currentLst)>2):
			self.canvas.create_line(self.currentLst[-1][0], self.currentLst[-1][1],self.currentLst[0][0],self.currentLst[0][1])
			(N,S,W,E) = self.findBoundaryPoint()
			y = N
			x = W
			h = S-N
			w = E-W
			print y,x,h,w
			print N,W,S,E
			crop_img = self.cv2Img[y:y+h,x:x+w]
			cv2.imwrite("test.jpg",crop_img)
			self.objLst.append(self.currentLst)
			self.currentLst=[]

	def findBoundaryPoint(self):
		N = -1
		S = -1
		W = -1
		E = -1
		print "findBoundaryPoint len(self.currentLst) = ", len(self.currentLst)
		for i in range(0,len(self.currentLst)):
		    if(self.currentLst[i][0] > E):
			E = self.currentLst[i][0]
		    if(self.currentLst[i][1] > S):
			S = self.currentLst[i][1]
		    if(self.currentLst[i][0] < W or W == -1):
			W = self.currentLst[i][0]
		    if(self.currentLst[i][1] < N or N == -1):
			N = self.currentLst[i][1]
		return (N,S,W,E)
	'''
	
root = TK.Tk()
root.title("label")
root.geometry("700x600")
app = Gui(root)
root.mainloop()