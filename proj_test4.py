import cv2
import cv2.cv as cv
import numpy as np
import ImageDraw
import Tkinter as TK
import tkSimpleDialog
import ImageTk as iTK   
import Image as i
import math

np.set_printoptions(threshold=np.nan)

class Gui(TK.Frame):    
        def __init__(self,master):
                TK.Frame.__init__(self,master)
                self.grid()
                self.objLst = []
                self.currentLst = []
                self.warpPosition = []
                self.create_widgets()           
                self.projectingXYZ = []
                self.projectingRGB = []
                                
                self.cameraPosition = np.array([325.0, 250.0, -2.0])
                self.rotationAxis = np.array([0, 1, 0])
                self.rotationDegree = 0

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
                self.canvas.bind_all('<KeyPress>',self.move)
                self.button=TK.Button(self)
                self.button["text"]="join"
                self.button["command"]=self.joinPoints
                self.button.grid()
                self.button=TK.Button(self)
                self.button["text"]="Add polygon"
                self.button["command"]=self.addPolygon
                self.button.grid()
                self.button=TK.Button(self)
                self.button["text"]="Perspective Projection"
                self.button["command"]=self.movingPerspectiveProject
                self.button.grid()
                
        def move(self,event):
                isControlKey = True
                if event.keysym=='Right':
                        self.cameraPosition[0]+=0.5
                elif event.keysym=='Left':
                        self.cameraPosition[0]-=0.5
                elif event.keysym=='Up':
                        self.cameraPosition[2] +=10
                elif event.keysym=='Down':
                        self.cameraPosition[2]-=10
                elif event.keysym=='d':
                        self.rotationDegree-=0.01
                elif event.keysym=='a':
                        self.rotationDegree+=0.01
                else:
                        isControlKey = False
                if isControlKey:
                        self.movingPerspectiveProject()
                
        def movingPerspectiveProject(self):
	        newPoints = np.array(self.projectingXYZ)
	        currentPointsColor = np.array(self.projectingRGB)
	        
	        newImage = i.new('RGB', (650, 500), "black") # create a new black image
	        pixels = newImage.load() # create the pixel map
	
	        newPoints = np.array(newPoints)
	    
	        # Perform Perspective Projection
	        #qp = self.quatmult(self.getQuaternionQ(self.rotationDegree, self.rotationAxis), [0, self.cameraPosition[0], self.cameraPosition[1], self.cameraPosition[2]])
	
	        #qpqPrime = self.quatmult(qp, self.getQuaternionQPrime(self.getQuaternionQ(self.rotationDegree, self.rotationAxis)))
	        self.cameraPosition = [self.cameraPosition[0], self.cameraPosition[1], self.cameraPosition[2]]
	        radian = math.radians(self.rotationDegree)
	        rotation1 = np.array([math.cos(radian/2),math.sin(radian/2)*0,math.sin(radian/2)*1,math.sin(radian/2)*0])
	        quatmat_1 = self.quat2rot(rotation1)  #(self.getQuaternionQ(self.rotationDegree, self.rotationAxis))
	           
	        camK = [quatmat_1[2,0],quatmat_1[2,1],quatmat_1[2,2]]
	        for j in xrange(len(newPoints)):
	            if np.dot((newPoints[j]-self.cameraPosition),camK)>0:
	                image1_u, image1_v = self.perspectiveProjection(newPoints[j], self.cameraPosition, quatmat_1[0], quatmat_1[1], quatmat_1[2])
	
	                if (325 + image1_u >= 0 and 325 <= 650) and (250 + image1_v >= 0 and 250 + image1_v <= 500):
	                
	                    pixels[325 + image1_u, 250 + image1_v] = (currentPointsColor[0, j], currentPointsColor[1, j], currentPointsColor[2, j], currentPointsColor[3, j])
	
	        newImage = np.array(newImage)
		cv2.imshow("test", newImage)
               
        def addPolygon(self):
                temp = np.array(np.transpose(np.matrix(np.array(self.warpPosition))))
		newPosition = np.zeros([2, len(temp[0, :])])		
		newPosition[0, :] = temp[0, :]
		newPosition[1, :] = temp[1, :]
		
                temp = np.array(np.transpose(np.matrix(np.array(self.currentLst))))
                currentPosition = np.ones([3, len(temp[0, :] - 1)])
                currentPosition[0, :] = temp[0, :]
                currentPosition[1, :] = temp[1, :]

                homographyMatrix = self.findHomography(newPosition, currentPosition)

                print "homographyMatrix = \n", homographyMatrix
                
                # Perform transformation using the Homograhy matrix
   		left = int(newPosition[0, :].min())
                right = int(newPosition[0, :].max())
                top = int(newPosition[1, :].min())
                bottom = int(newPosition[1, :].max())
                
                newPosition = []
                
                self.drawWarpedImage()
                
                outImage = i.open("warpedImage.png").convert("RGBA")
                pixels = outImage.load()
                
                for y in range(top, bottom + 1):
                        for x in range(left, right + 1):
                        	if pixels[x, y][3] != 0:
					newPosition.append((x, y))
						
                currentPosition = self.homographyWarp(newPosition, homographyMatrix)
                newPosition = self.getZ(np.transpose(np.matrix(np.array(newPosition))))
                                                            
                # Reading colours
                outImage = i.open("out.png").convert("RGBA")
                pixels = outImage.load()
                
                currentPointsColor = []
                currentPosition = np.array(currentPosition)
                for j in range(0, len(currentPosition[0, :])):
                	currentPointsColor.append((pixels[currentPosition[0, j], currentPosition[1, j]][0], pixels[currentPosition[0, j], currentPosition[1, j]][1], pixels[currentPosition[0, j], currentPosition[1, j]][2],pixels[currentPosition[0, j], currentPosition[1, j]][3]))

                newPosition = np.transpose(np.matrix(np.array(newPosition)))
                currentPointsColor = np.transpose(np.matrix(np.array(currentPointsColor)))

                # Add on new coordinates, with its RGB
                if len(self.projectingXYZ) == 0:
                	self.projectingXYZ = newPosition
                else:
                	self.projectingXYZ = np.concatenate((self.projectingXYZ, np.array(newPosition)))
                	
                if len(self.projectingRGB) == 0:
			self.projectingRGB = currentPointsColor
		else:
			self.projectingRGB = np.concatenate((self.projectingRGB, np.array(currentPointsColor)))
                
                self.currentLst = []
                self.warpPosition = []
        
        def getQuaternionQ(self, rotationDegree, rotationAxis):
                q0 = np.cos(np.radians(rotationDegree / 2))
                q1 = np.sin(np.radians(rotationDegree / 2))
                q2 = np.sin(np.radians(rotationDegree / 2))
                q3 = np.sin(np.radians(rotationDegree / 2))
                q = np.array([q0, q1 * rotationAxis[0], q2 * rotationAxis[1], q3 * rotationAxis[2]])
                
                return q
                
        def getQuaternionQPrime(self, q):
                return np.array([q[0], q[1] * self.rotationAxis[0] * -1, q[2] * self.rotationAxis[1] * -1, q[3] * self.rotationAxis[2] * -1])
        
        def quatmult(self,q1, q2):
                # quaternion multiplication
                q1q2 = [0, 0, 0, 0]
                
                q1q2[0] = (q1[0] * q2[0]) - (q1[1] * q2[1]) - (q1[2] * q2[2]) - (q1[3] * q2[3])
                q1q2[1] = (q1[0] * q2[1]) + (q1[1] * q2[0]) + (q1[2] * q2[3]) - (q1[3] * q2[2])
                q1q2[2] = (q1[0] * q2[2]) - (q1[1] * q2[3]) + (q1[2] * q2[0]) + (q1[3] * q2[1])
                q1q2[3] = (q1[0] * q2[3]) + (q1[1] * q2[2]) - (q1[2] * q2[1]) + (q1[3] * q2[0])
                
                return q1q2
        
        def quat2rot(self, q):
                rotation = np.zeros([3, 3])
                rotation[0,0] = (q[0] * q[0]) + (q[1] * q[1]) - (q[2] * q[2]) - (q[3] * q[3])
                rotation[0,1] = 2 * ((q[1] * q[2]) - (q[0] * q[3]))
                rotation[0,2] = 2 * ((q[1] * q[3]) + (q[0] * q[2]))
                rotation[1,0] = 2 * ((q[1] * q[2]) + (q[0] * q[3]))
                rotation[1,1] = (q[0] * q[0]) + (q[2] * q[2]) - (q[1] * q[1]) - (q[3] * q[3])
                rotation[1,2] = 2 * ((q[2] * q[3]) - (q[0] * q[1]))
                rotation[2,0] = 2 * ((q[1] * q[3]) - (q[0] * q[2]))
                rotation[2,1] = 2 * ((q[2] * q[3]) + (q[0] * q[1]))
                rotation[2,2] = (q[0] * q[0]) + (q[3] * q[3]) - (q[1] * q[1]) - (q[2] * q[2])
        
                return np.matrix(rotation)
                
        def perspectiveProjection(self, s_p, t_f, i_f, j_f, k_f):
                s_p = np.matrix(s_p)
                t_f = np.matrix(t_f)
                i_f = np.matrix(i_f)
                j_f = np.matrix(j_f)
                k_f = np.matrix(k_f)
                                
                image_u = (((1 * (s_p - t_f)) * np.transpose(i_f)) / ((s_p - t_f) * np.transpose(k_f))) * 1 + 0
                image_v = (((1 * (s_p - t_f)) * np.transpose(j_f)) / ((s_p - t_f) * np.transpose(k_f))) * 1 + 0
                
                return image_u.item(), image_v.item()

                
        def getZ(self, newPoints):
		temp = np.array(self.warpPosition) #user input

		#for j in range(0,temp.shape[0]):
		#       print "warpPosition Point=", j, "in form of (x,y,z): (",temp[j,0], ",", temp[j,1] , "," , temp[j,2] , ")"

		# To get normal Vector (orthogonal) e.g. need two vectors
		# PointA = temp[0,0], temp[0,1], temp[0,2]
		# PointB = temp[1,0], temp[1,1], temp[1,2]
		# PointC = temp[2,0], temp[2,1], temp[2,2]

		# Point CB = B - C 
		PointCB_xaxis = temp[1,0] - temp[2,0]
		PointCB_yaxis = temp[1,1]- temp[2,1]
		PointCB_zaxis = temp[1,2] - temp[2,2]
		print "PointCB_xaxis", PointCB_xaxis, "PointCB_yaxis: ",PointCB_yaxis, "PointCB_zaxis: ",PointCB_zaxis

		# PointAB = B - A
		PointAB_xaxis = temp[1,0] - temp[0,0]
		PointAB_yaxis = temp[1,1]- temp[0,1]
		PointAB_zaxis = temp[1,2] - temp[0,2]
		print "PointAB_xaxis", PointAB_xaxis, "PointAB_yaxis: ",PointAB_yaxis, "PointAB_zaxis: ",PointAB_zaxis

		#To get normal vector (orthogonal), use formula.
		NormalVector_i = ((PointAB_yaxis*PointCB_zaxis)-(PointAB_zaxis*PointCB_yaxis))
		NormalVector_j = -1 * ((PointAB_xaxis*PointCB_zaxis)-(PointAB_zaxis*PointCB_xaxis))
		NormalVector_k = ((PointAB_xaxis*PointCB_yaxis)-(PointAB_yaxis*PointCB_xaxis))

		i_int = NormalVector_i * temp[1,0]
		j_int = NormalVector_j * temp[1,1]
		k_int = NormalVector_k * temp[1,2]
		# allInt = i_int + j_int - k_int

		a = np.matrix([NormalVector_i, NormalVector_j, NormalVector_k])
		b = np.matrix([temp[0,0], temp[0,1], temp[0,2]])
		c = a * np.transpose(b)

		returnNewPoints = []
		#(NormalVector_i * x) + (NormalVector_j * y) + (NormalVector_k * z) = - allInt
		#z = (-allInt - (NormalVector_i * x) - (NormalVector_j * y))/NormalVector_k
		for j in range(0,newPoints.shape[1]):
			# z = (-allInt - (NormalVector_i *  newPoints[0,j] ) - (NormalVector_j * newPoints[1,j])) / (NormalVector_k)
			# print "get Z New Points=", j, "in form of (x,y): (", newPoints[0,j], ",", newPoints[1,j],")"
			# print "get Z New Points=", j, "in form of (x,y,z): (", newPoints[0,j], ",", newPoints[1,j], ",", z ,")"

			z = ((NormalVector_i * newPoints[0,j]) + (NormalVector_j * newPoints[1,j]) - c) / (-1 * NormalVector_k)
			row = np.array([[newPoints[0,j], newPoints[1,j], z]])

			if j == 0:
				returnNewPoints = row
			else:
				returnNewPoints = np.concatenate((returnNewPoints, row))

		returnNewPoints = np.transpose(np.matrix(returnNewPoints))
		return returnNewPoints

		#NormalVector =  NormalVector_i*i - NormalVector_j*j + NormalVector_k*k, where i = x - temp[0,0] and etc.
		#equationOfPlane = NormalVector_i * x + NormalVector_j * y + NormalVector_k * z = - allInt
	        
                #print "NormalVector= ", NormalVector
                
        def homographyWarp(self, currentPoints, homographyMatrix):
        	currentPoints = np.transpose(np.matrix(currentPoints))
        	
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
                # print "newPoints = \n", newPoints
                
                return newPoints
        
        def findHomography(self, currentPosition, newPosition):
                if currentPosition.shape[1] != newPosition.shape[1]:
                        return "Points matrices different in sizes."
                
                # if newPosition.shape[0] != 2:
                #        return "Points matrices must have two rows."

                # if newPosition.shape[1] < 4:
                #        return "Need at least 4 matching points." 
        
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
                
        def drawWarpedImage(self):
        	list = []
        	for j in range(0, len(self.warpPosition)):
        		list.append(np.array([self.warpPosition[j][0], self.warpPosition[j][1]]))
        		
		if(len(list)>2):			
			im = i.new('RGBA', (650, 500), "black") # create a new black image

			# convert to numpy (for convenience)
			imArray = np.asarray(im)

			# create mask
			polygon = []
			for j in range(0, len(list)):
				polygon.append((list[j][0], list[j][1]))

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
			newIm.save("warpedImage.png")

			self.objLst.append(self.currentLst)
                        # self.currentLst=[]
        
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
        
root = TK.Tk()
root.title("label")
root.geometry("700x600")
app = Gui(root)
root.mainloop()
