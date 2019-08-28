from tkinter import filedialog
from tkinter import *
#from Tkinter import Tk
from tkinter.filedialog import askopenfilename,askdirectory
from imageio import imread
import cv2
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import gaussian_filter

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


def negative(img, dmin=0, dmax=255):
    return np.uint8(np.clip(-img + dmax, dmin, dmax - 1))

def mean_filter(img, N):
    kernel = np.ones((N,N))/(N*N)
    return signal.convolve2d(img, kernel, boundary='symm', mode='same')

def pers_filter(img,array):
    return signal.convolve2d(img, array, boundary='symm', mode='same')


a_min = 0    # the minimial value of the paramater a
a_max = 1   # the maximal value of the paramater a
a_init = 1.0   # the value of the parameter a to be used initially, when the graph is created
ySlider= 0.22
yFiltro=0.23


check1_ax = plt.axes([0.001, 0.6, 0.1, 0.035])
check1_ax.set_frame_on(False)

check2_ax = plt.axes([0.001, 0.5, 0.1, 0.035])
check2_ax.set_frame_on(False)


sin_ax = plt.axes([0.01, 0.3, 0.5, 0.65])

slider_R = plt.axes([0.1, ySlider+0.02, 0.35, 0.01])
slider_G = plt.axes([0.1, ySlider-0.01, 0.35, 0.01])
slider_B = plt.axes([0.1,ySlider-0.04, 0.35, 0.01])
slider_a = plt.axes([0.1,ySlider-0.07, 0.35, 0.01])
slider_b = plt.axes([0.1,ySlider-0.10, 0.35, 0.01])
slider_g = plt.axes([0.1,ySlider-0.13, 0.35, 0.01])
slider_Al = plt.axes([0.1,ySlider-0.16, 0.35, 0.01])

histo = plt.axes([0.51, 0.3, 0.45, 0.65])

Cimagen = plt.axes([0.001, 0.9, 0.1, 0.035])#img1
Cimagen2 = plt.axes([0.001, 0.8, 0.1, 0.035])

nega = plt.axes([0.5,ySlider, 0.1, 0.035])
toGray = plt.axes([0.5,ySlider-0.05, 0.1, 0.035])
toRGB = plt.axes([0.5,ySlider-0.10, 0.1, 0.035])
Blends =plt.axes([0.5,ySlider-0.15, 0.1, 0.035])
filsobel = plt.axes([0.62, ySlider, 0.1, 0.035])
filave = plt.axes([0.62, ySlider-0.05, 0.1, 0.035])
filgauss = plt.axes([0.62, ySlider-0.10, 0.1, 0.035])
filPersonal = plt.axes([0.74,ySlider-0.16, 0.1, 0.035])
histEqu = plt.axes([0.62, ySlider-0.15, 0.1, 0.035])

wardar = plt.axes([0.89,ySlider, 0.1, 0.035])#fila1

f11 = plt.axes([0.74,yFiltro, 0.04, 0.04])#fila1
f12 = plt.axes([0.786,yFiltro, 0.04, 0.04])
f13 = plt.axes([0.831,yFiltro, 0.04, 0.04])

f21 = plt.axes([0.74,yFiltro-0.05, 0.04, 0.04])
f22 = plt.axes([0.786,yFiltro-0.05, 0.04, 0.04])
f23 = plt.axes([0.831,yFiltro-0.05, 0.04, 0.04])

f31 = plt.axes([0.74,yFiltro-0.1, 0.04, 0.04])
f32 = plt.axes([0.786,yFiltro-0.1, 0.04, 0.04])
f33 = plt.axes([0.831,yFiltro-0.1, 0.04, 0.04])


check = imread('./check.png')
x = imread('./x.png')
img = imread('./jelly.jpg')
img2 = imread('./leon.jpg')

guardar=img

img1B= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2B= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.rcParams.update({'font.size': 17})
plt.axes(check1_ax)
plt.imshow(x)
plt.axes(check2_ax)
plt.imshow(x)
plt.axes(sin_ax)
plt.title('PhotoShop')
plt.imshow(img)
plt.axes(histo)
plt.title('Histograma')
plt.hist(img.ravel(), 256, [0, 256])



R_slider = Slider(slider_R,'Red',a_min,a_max,valinit=a_init)
G_slider = Slider(slider_G,'Green',a_min,a_max,valinit=a_init)
B_slider = Slider(slider_B,'Blue',a_min,a_max,valinit=a_init)
a_slider = Slider(slider_a,'Contrast',-15,15,valinit=a_init)
b_slider = Slider(slider_b,'Brightness',-100,100,0)
g_slider = Slider(slider_g,'Gamma Corr',-5,5,valinit=a_init)
Al_slider = Slider(slider_Al,'Alpha',a_min,a_max,valinit=a_init)


#axprev = plt.axes([0.7, 0.05, 0.1, 0.075])

carga = Button(Cimagen, 'Subir Img 1')
carga2= Button(Cimagen2,'Subir Img 2')
bfilsobel = Button(filsobel, 'Fil Sobel')
bfilave = Button(filave, 'Fil Ave')
bfilgauss = Button(filgauss, 'Fil Gauss')
bnega = Button(nega, 'Negativo')
btoGray = Button(toGray, 'RGB to Gray')
btoRGB = Button(toRGB, 'Gray to RGB')
bfilPersonal = Button(filPersonal, 'Fil Personal')
bBlend= Button(Blends,'Linear Blend')
bHistEq = Button(histEqu,'Histogram Eq')

bwardar = Button(wardar,'Guardar')

PersonalFil=[]
PersonalFil.extend((TextBox(f11, '', initial="1"),TextBox(f12, '', initial="2"),TextBox(f13, '', initial="3"),
                   TextBox(f21, '', initial="4"),TextBox(f22, '', initial="5"),TextBox(f23, '', initial="6"),
                   TextBox(f31, '', initial="7"),TextBox(f32, '', initial="8"),TextBox(f33, '', initial="9")))

def update(a):    
    global guardar
    plt.Axes.clear(histo)    
    img_red = (a_slider.val*img +b_slider.val )* [R_slider.val, G_slider.val, B_slider.val]
    img_red = np.uint8(np.clip(((img_red / 255) ** (1.0 / g_slider.val)) * 255, 0, 255 - 1))
    img_red = np.clip(img_red, 0.0, 255.0).astype(np.uint8)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
#    cv2.equalizeHist(histo)
    
def subir(a):
    global img,img1B,guardar
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    img1B= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    print(filename)
    img = imread(filename)
    plt.axes(sin_ax)
    plt.title('PhotoShop')
    plt.imshow(img)
    guardar=img
    plt.axes(check1_ax)
    plt.imshow(check)
    plt.show()
    
def subir2(a):
    global img2,img2B
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#    print(filename)
    img2 = imread(filename)
    img2B= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#    plt.axes(sin_ax)
#    plt.title('PhotoShop')
#    plt.imshow(img2)    
    plt.axes(check2_ax)
    plt.imshow(check)
    plt.show()
    
def RGBtoGray(a):
    global guardar
    plt.Axes.clear(histo)
    imgB= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.axes(sin_ax)
    plt.imshow(imgB, cmap = 'gray')
    guardar=imgB
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img2.ravel(), 256, [0, 256])
    plt.show()
    
def GraytoRGB(a):  
    global guardar
    plt.Axes.clear(histo)
    plt.axes(sin_ax)
    plt.imshow(img)
    guardar=img
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img.ravel(), 256, [0, 256])        
    plt.show()
    
def updNeg (a):
    global guardar
    plt.Axes.clear(histo)
    img_red = negative(img,0,255)
    #img_red = np.clip(img_red, 0.0, 255.0).astype(np.uint8)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()

def personal(a):
    array = np.ones(9)
    a=0
    for x in PersonalFil:
        array[a]=x.text.rstrip()
        a=a+1
        #print (x.text.rstrip()) #LEE LOS VALORES DEL FILTRO PERSONALIZADO
    #print(array)
    fil=np.resize(array,(3,3))
    return fil
    
def Blend(a):
    global guardar
    plt.Axes.clear(histo)
    img_red = np.uint8(np.clip((1.0 - Al_slider.val) * img + Al_slider.val * img2, 0, 254))
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()
    
def lBlend(a):
    global guardar
    plt.Axes.clear(histo)
    pattern = np.array([np.arange(img1B.shape[1])]).repeat(img1B.shape[0], 0) / img1B.shape[1]
    img_red = np.uint8(np.clip((-pattern+1)*img1B + pattern*img2B, 0, 254))
    plt.axes(sin_ax)
    plt.imshow(img_red,cmap='gray')
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()
    
def filPers(a):
    global guardar
    plt.Axes.clear(histo)
    fil = personal(1)
    print(fil)
    img_red = pers_filter(img1B,fil)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()
    
def filAv(a):
    global guardar
    plt.Axes.clear(histo)
    img_red = mean_filter(img1B,3)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()
    
def filGauss(a):
    global guardar
    plt.Axes.clear(histo)
    img_red = gaussian_filter(img, 5)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(img_red))
    guardar=img_red
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()

def filSob(a):
    global guardar
    plt.Axes.clear(histo)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(gx))
    plt.axes(histo)
    plt.imshow(np.uint8(gy))
    guardar=gx
    #plt.hist(img_red.ravel(), 256, [0, 256])
    plt.show()
    
def HistoEq(a):
    global guardar
    #plt.Axes.clear(histo)
    imgGrayEqu = cv2.equalizeHist(img1B)
    plt.axes(sin_ax)
    plt.imshow(np.uint8(imgGrayEqu))
    guardar=imgGrayEqu
    plt.axes(histo)
    plt.title('Histograma')
    plt.hist(imgGrayEqu.ravel(), 256, [0, 256])
    plt.show()
#def guarda(a):
#    filename = askdirectory()
###    filename = filedialog.askdirectory()
#    cv2.imwr
#    print(filename)


R_slider.on_changed(update)
G_slider.on_changed(update)
B_slider.on_changed(update)
a_slider.on_changed(update)
b_slider.on_changed(update)
g_slider.on_changed(update)
Al_slider.on_changed(Blend)

carga.on_clicked(subir)
carga2.on_clicked(subir2)
bnega.on_clicked(updNeg)
bfilave.on_clicked(filAv)
bfilgauss.on_clicked(filGauss)
bfilsobel.on_clicked(filSob)
btoGray.on_clicked(RGBtoGray)
btoRGB.on_clicked(GraytoRGB)
bfilPersonal.on_clicked(filPers)
bBlend.on_clicked(lBlend)     
bHistEq.on_clicked(HistoEq)

bwardar.on_clicked(guarda)

plt.show()


