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
    final_image = output_image

    # Tranformations on image since the dataset was mirrored and rotated 90 deg
    final_image = PIL.ImageOps.invert(final_image)
    final_image = PIL.ImageOps.mirror(final_image)
    final_image = final_image.rotate(90)
    final_image = (np.array(final_image.resize(size, PIL.Image.ANTIALIAS)))
    final_image = np.expand_dims(np.reshape(final_image, (784,)), axis=0)

    model = tf.keras.models.load_model('letter_digit_recognizer.model')
    model_prediction = model.predict(final_image / 255)

    if np.argmax(model_prediction) < 10:
        print(np.argmax(model_prediction))
    elif np.argmax(model_prediction) >= 10 and np.argmax(model_prediction) <= 35:
        print(chr(np.argmax(model_prediction) - 10 + 65))
    else:
        print(chr(np.argmax(model_prediction) - 36 + 97))


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