import cv2
import numpy as np
import ImageDraw
import Tkinter as TK
import tkSimpleDialog
import ImageTk as iTK	
import Image as i

class Gui(TK.Frame):
    
	def __init__(self,master):
	        TK.Frame.__init__(self,master)
	        self.grid()
	        self.objLst = []
	        self.currentLst = []
	        self.create_widgets()
	        self.cameraPosition = np.matrix(np.zeros([1,3]))

	def clickCallback(self,event):
		print (event.x,event.y)
		depth = tkSimpleDialog.askinteger("Depth", "Please enter the depth.")
		if depth != None:
	         	print depth
			self.canvas.create_rectangle(event.x-1,event.y-1,event.x+1,event.y+1)
			self.currentLst.append(np.array([event.x, event.y, depth]))
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
			self.currentLst=[]
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