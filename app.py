import numpy as np
import gradio as gr

def sepia(input_img):
    # Convert input image to float32 and apply the sepia filter
    input_img = input_img.astype(np.float32)
    sepia_filter = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255)  # Ensure pixel values are in the correct range
    return sepia_img.astype(np.uint8)

demo = gr.Interface(fn=sepia, inputs=gr.Image(type="numpy"), outputs="image")

demo.launch(share=True)
