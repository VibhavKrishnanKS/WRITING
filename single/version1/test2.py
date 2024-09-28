import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tkinter as tk

model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_and_predict():
    img = Image.new("L", (200, 200), 255) 
    draw = ImageDraw.Draw(img)
    
    for item in canvas.find_all():
        x0, y0, x1, y1 = canvas.coords(item)
        draw.ellipse([x0 - 5, y0 - 5, x1 + 5, y1 + 5], fill=0)

    img = img.resize((28, 28), Image.LANCZOS)
    
    img = ImageOps.invert(img)
    
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)   

    prediction = model.predict(img_array)

    print("Raw model output:", prediction)  

    predicted_class = np.argmax(prediction)
    
    predicted_digit = predicted_class  

    prediction_label.config(text=f'Predicted Digit: {predicted_digit}')

def clear_canvas():
    canvas.delete("all")

def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", outline="black", width=10)

root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

canvas.bind("<B1-Motion>", draw)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

predict_button = tk.Button(button_frame, text="Predict Digit", command=preprocess_and_predict)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.grid(row=0, column=1, padx=10)

prediction_label = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 16))
prediction_label.pack(pady=10)

root.mainloop()
