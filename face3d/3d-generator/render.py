import ctypes
import sys
import time 
import os
sys.path.append('..')

import numpy as np
import pyglet
from pyglet.gl import *

from pywavefront import visualization
import pywavefront
import cv2 
import imutils
import dlib
from imutils import face_utils

rotation = 0
path = sys.argv[1]
files = os.listdir(path)
for file in files:
    if file.split('.')[-1]=='obj':
        path = path+file
#print(path)
meshes = pywavefront.Wavefront(path)
output = './output/'+path.split('/')[2]+'/'
#print(output)
if not os.path.isdir('./output/'):
    os.mkdir('./output/')
if not os.path.isdir(output):
    os.mkdir(output)
window = pyglet.window.Window()
lightfv = ctypes.c_float * 4
global n
n = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./3d-generator/generator/shape_predictor_68_face_landmarks.dat')

@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    global n
    window.clear()
    glLoadIdentity()
    
    r = np.random.rand(3)*10-5
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(r[0],r[1],r[2],np.random.rand()))
    r = np.random.rand(3)*10-5
    glLightfv(GL_LIGHT1, GL_POSITION, lightfv(r[0],r[1],r[2],np.random.rand()))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHTING)

    glTranslated(0, 0, -10)
    glRotatef(np.random.rand()*90-45, 0.0, 1.0, 0.0)
    glRotatef(np.random.rand()*60-30, 1.0, 0.0, 0.0)
    glRotatef(np.random.rand()*40-20, 0.0, 0.0, 1.0)

    glEnable(GL_LIGHTING)
    
    visualization.draw(meshes)
    #######################################################
    pyglet.image.get_buffer_manager().get_color_buffer().save(output + 'temp.png')
    #######################################################
    image = cv2.imread(output + 'temp.png')
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        crop_img = image[y:y+h, x:x+w]
        cv2.imwrite(output + str(n) + '.png', crop_img)
        n += 1
        print(output,n)
    ########################################################
    if n>int(sys.argv[2]):
        os.remove(output + 'temp.png')
        exit()

def update(dt):
    global rotation
    rotation += 90.0 * dt*0.2

    if rotation > 720.0:
        rotation = 0.0


pyglet.clock.schedule(update)
pyglet.app.run()
