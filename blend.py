from tkinter import filedialog
from tkinter import *
#from Tkinter import Tk
from tkinter.filedialog import askopenfilename
from imageio import imread
import cv2
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import matplotlib.pyplot as plt

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

def blend(img1, img2, alpha, dmin=0, dmax=255):
    return np.uint8(np.clip((1.0-alpha)*img1 + alpha*img2, dmin, dmax-1))

def linearBlend(img1, img2, dmin=0, dmax=255):
    pattern = np.array([np.arange(img1.shape[1])]).repeat(img1.shape[0],0)/img1.shape[1]
    return np.uint8(np.clip((-pattern+1)*img1 + pattern*img2, dmin, dmax-1))

def subir(a):
    global img
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#    print(filename)
    img = imread(filename)
    plt.axes(img1a)
    plt.title('PhotoShop')
    plt.imshow(img)
    plt.show()
    
def subir2(a):
    global img2
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#    print(filename)
    img2 = imread(filename)
    plt.axes(img2a)
    plt.title('PhotoShop')
    plt.imshow(img2)
    plt.show()

def updateAlpha():
    res = blend(img,img2,a_slider.val,0,255)
    plt.axes(res)
    plt.imshow(np.uint8(res))
    plt.show()
    #plt.imshow(blend(img,img2,a_slider.val,0,255))    
def updateAlphaL():
    res = linearBlend(img,img2,0,255)
    plt.axes(res)
    plt.imshow(np.uint8(res))
    plt.show()

a_min = 0    # the minimial value of the paramater a
a_max = 1   # the maximal value of the paramater a
a_init = 1.0   # the value of the parameter a to be used initially, when the graph is created
ySlider= 0.22
yFiltro=0.23

img1a = plt.axes([0.01, 0.3, 0.5, 0.65])
img2a = plt.axes([0.51, 0.3, 0.45, 0.65])
res = plt.axes([0.101, 0.3, 0.45, 0.65])
slider_a = plt.axes([0.1, ySlider+0.02, 0.35, 0.01])
slider_aL = plt.axes([0.1, ySlider-0.01, 0.35, 0.01])

carga1 = plt.axes([0.5,ySlider, 0.1, 0.035])
carga2 = plt.axes([0.62, ySlider, 0.1, 0.035])

img = imread('./jelly.jpg')
img2 = imread('./jelly.jpg')



a_slider = Slider(slider_a,'Alpha Blend',a_min,a_max,valinit=a_init)
#aL_slider = Slider(slider_aL,'Alpha Linear Blend',a_min,a_max,valinit=a_init)

carga = Button(carga1, 'Subir Img 1')
carga2 = Button(carga2, 'Subir Img 2')

a_slider.on_changed(updateAlpha)
#aL_slider.on_changed(updateAlphaL)
    
carga.on_clicked(subir)
carga2.on_clicked(subir2)