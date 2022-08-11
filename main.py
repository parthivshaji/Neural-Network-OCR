from typing import final
from PIL import ImageTk, Image, ImageDraw
import PIL
import PIL.ImageOps
from tkinter import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

width = 200  # canvas width
height = 200 # canvas height
center = height//2
white = (255, 255, 255) # canvas back

def detectLetter():
    size = (28, 28)
    #final_image = output_image.convert('L')
    final_image = output_image
    final_image = PIL.ImageOps.invert(final_image)
    final_image = PIL.ImageOps.mirror(final_image)
    final_image = final_image.rotate(90)
    final_image = (np.array(final_image.resize(size, PIL.Image.ANTIALIAS)))
    final_image = np.expand_dims(np.reshape(final_image, (784,)), axis=0)
    print(np.shape(final_image))
    char_model = tf.keras.models.load_model('char_recognizer.model')
    digit_model = tf.keras.models.load_model('digit_recognizer.model')

    char_model_prediction = char_model.predict(final_image / 255)
    digit_model_prediction = digit_model.predict(final_image / 255)

    if (np.max(char_model_prediction) > np.max(digit_model_prediction)):
        print("letter",chr(np.argmax(char_model_prediction) + 64))
    else:
        print("number",np.argmax(digit_model_prediction))
        print(digit_model_prediction)

def clearCanvas():
    global output_image
    global draw
    canvas.delete("all")
    output_image = PIL.Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(output_image)
    canvas.pack(expand=YES, fill=BOTH)
    canvas.bind("<B1-Motion>", paint)

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=10)
    draw.line([x1, y1, x2, y2],fill="black",width=10)

master = Tk()

# create a tkinter canvas to draw on
canvas = Canvas(master, width=width, height=height, bg='white')
canvas.pack()

# create an empty PIL image and draw object to draw on
output_image = PIL.Image.new("L", (width, height), 255)
draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button=Button(text="Detect Character",command=detectLetter)
button.pack()

button=Button(text="Clear Canvas",command=clearCanvas)
button.pack()

master.mainloop()