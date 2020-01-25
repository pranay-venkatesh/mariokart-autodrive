"""

MarioKart bot for automatic drive.
Used tensorflow and opencv

Made by Pranay Venkatesh aka supremefiend101


It proceeds by imitation. The goal is to try and copy my moveset in the game.
The input to the neuralnet is the live screenshot from Super Mario Kart and the output is one of : move forward, turn left, turn right, reverse
By observing which key I press during the training phase, the neural net trains itself to try and do the same thing in similar situations.


"""

import numpy as np
import cv2
import pyscreenshot as pysc
import pyautogui
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation = tf.nn.softmax))


# Defining the output functions of the neural network.
def forward():
    pyautogui.keyDown('z')

def left():
    pyautogui.keyDown('z')
    pyautogui.keyDown('left')

def right():
    pyautogui.keyDown('z')
    pyautogui.keyDown('right')

def reverse():
    pyautogui.keyDown('x')

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
    except:
        pass
while (True):
    # Image processing phase.
    screen = pysc.grab(bbox = (55, 80, 690, 590))
    screen = np.array(screen)
    cannied = cv2.Canny(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), threshold1 = 150, threshold2 = 250)
    lines = cv2.HoughLinesP (cannied, 1, np.pi/180, 180, 20, 15)
    draw_lines (cannied, lines)
    cv2.imshow('gamefeed', cannied)
    
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break


