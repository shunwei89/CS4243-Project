import cv2
import Tkinter as TK
import ImageTk as iTK
import Image as i
import numpy
import ImageDraw

class Gui(TK.Frame):
    
    def __init__(self,master):
        TK.Frame.__init__(self,master)
        self.grid()
        self.objLst = []
        self.currentLst = []
        self.lst = []
        self.create_widgets()

    def clickCallback(self,event):
         print (event.x,event.y)
         self.canvas.create_rectangle(event.x-1,event.y-1,event.x+1,event.y+1)
         self.currentLst.append(event)
         if len(self.currentLst)>1:
             self.canvas.create_line(self.currentLst[-2].x, self.currentLst[-2].y,self.currentLst[-1].x,self.currentLst[-1].y)

    def create_widgets(self):
        # read image as RGB and add alpha (transparency)
        im = i.open('project.jpeg').convert("RGBA")
               
        self.cv2Img = cv2.imread('project.jpeg')
        image = im.resize((650, 500), i.ANTIALIAS)
        self.img = iTK.PhotoImage(image)

        # convert to numpy (for convenience)
        imArray = numpy.asarray(im)

        self.canvas = TK.Canvas(self,width=650,height=500)
        self.canvas.create_image(325,250,imag=self.img)
        #self.canvas.itemconfigure(self.img,state=NORMAL)
        self.canvas.grid()
        self.canvas.bind("<Button 1>",self.clickCallback)
        self.button=TK.Button(self)
        #self.button["text"]="join"
        self.button["text"]="Generate Mask"
        #self.button["command"]=self.joinPoints
        self.button["command"]=self.generateMask
        self.button.grid()


    def generateMask(self):
        # create mask
        im = i.open('project.jpeg').convert("RGBA")
        # convert to numpy (for convenience)
        imArray = numpy.asarray(im)

        #polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
        maskIm = i.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(self.currentLst, outline=1, fill=1)
        mask = numpy.array(maskIm)
        '''
        # assemble new image (uint8: 0-255)
        newImArray = numpy.empty(imArray.shape,dtype='uint8')

        # colors (three first columns, RGB)
        newImArray[:,:,:3] = imArray[:,:,:3]

        # transparency (4th column)
        newImArray[:,:,3] = mask*255

        # back to Image from numpy
        newIm = Image.fromarray(newImArray, "RGBA")
        newIm.save("out.png")
        
       #COMMAND AWAY
       for i in range(len(self.currentLst)):
            s = (self.currentLst[i].x),(self.currentLst[i].y),"," 
            print s

        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
        mask = numpy.array(maskIm)
        




    def joinPoints(self):
        if(len(self.currentLst)>2):
             self.canvas.create_line(self.currentLst[-1].x, self.currentLst[-1].y,self.currentLst[0].x,self.currentLst[0].y)
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
        for i in range(0,len(self.currentLst)):
            if(self.currentLst[i].x > E):
                E = self.currentLst[i].x
            if(self.currentLst[i].y > S):
                S = self.currentLst[i].y
            if(self.currentLst[i].x < W or W == -1):
                W = self.currentLst[i].x
            if(self.currentLst[i].y < N or N == -1):
                N = self.currentLst[i].y
        return (N,S,W,E)

    '''

root = TK.Tk()
root.title("label")
root.geometry("700x600")
app = Gui(root)
root.mainloop()

