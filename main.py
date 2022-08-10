from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import tensorflow as tf
import numpy as np

width = 200  # canvas width
height = 200 # canvas height
center = height//2
white = (255, 255, 255) # canvas back

def detectLetter():
    global output_image
    size = (28, 28)
    output_image = output_image.convert('L')
    output_image = (np.array(output_image.resize(size, PIL.Image.ANTIALIAS)))
    output_image = [np.reshape(output_image, (784,))]
    char_model = tf.keras.models.load_model('char_recognizer.model')
    digit_model = tf.keras.models.load_model('digit_recognizer.model')
    char_model_prediction = char_model.predict(np.divide(output_image, 255))
    print("char",np.argmax(char_model_prediction))
    output_image = PIL.Image.new("RGB", (width, height), white)

def clearCanvas():
    global output_image
    canvas.delete("all")
    output_image = PIL.Image.new("RGB", (width, height), white)

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

master = Tk()

# create a tkinter canvas to draw on
canvas = Canvas(master, width=width, height=height, bg='white')
canvas.pack()

# create an empty PIL image and draw object to draw on
output_image = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button=Button(text="Detect Letter",command=detectLetter)
button.pack()

button=Button(text="Clear Canvas",command=clearCanvas)
button.pack()

master.mainloop()