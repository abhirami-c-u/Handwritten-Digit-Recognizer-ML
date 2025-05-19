
# Save this code as digit_gui_predictor.py and run it after training the model

import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("mnist_cnn_model.h5")

# Set up GUI window
window = Tk()
window.title("Handwritten Digit Recognizer")
window.resizable(0, 0)
canvas = Canvas(window, width=280, height=280, bg='white')
canvas.pack()

# Drawing tool
image1 = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=255)
    label_result.config(text="Draw a digit and click Predict")

def predict_digit():
    # Resize to 28x28 and invert colors
    image_resized = image1.resize((28, 28))
    image_resized = ImageOps.invert(image_resized)
    image_array = np.array(image_resized) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    pred = model.predict(image_array)
    digit = np.argmax(pred)
    label_result.config(text=f"Prediction: {digit}")

btn_predict = Button(window, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = Button(window, text="Clear", command=clear_canvas)
btn_clear.pack()

label_result = Label(window, text="Draw a digit and click Predict", font=("Helvetica", 16))
label_result.pack()

window.mainloop()
