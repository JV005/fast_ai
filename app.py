import os

# Install required packages
os.system("pip install fastai gradio torch")

__all__ = ['is_cat', 'learn', 'classify_image', 'image', 'label', 'examples', 'inf']

from fastai.vision.all import *
import gradio as gr
import pathlib

def is_cat(x): return x[0].isupper()

# pathlib.PosixPath = pathlib.WindowsPath  # Override PosixPath to use WindowsPath
learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    img = PILImage.create(img)  # Convert to fastai-compatible format
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))
    
image = gr.Image(type="pil")  # 'shape' is removed, Gradio automatically resizes if needed
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

inf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
inf.launch()
