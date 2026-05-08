import gradio as gr
import numpy as np

# Dummy prediction function
def bottleneck_prediction(cpu_score, gpu_score):

    result = (gpu_score - cpu_score) / 10

    return f"Predicted Bottleneck: {result:.2f}%"

interface = gr.Interface(
    fn=bottleneck_prediction,
    inputs=[
        gr.Number(label="CPU Score"),
        gr.Number(label="GPU Score")
    ],
    outputs="text",
    title="AI Bottleneck Predictor",
    description="Predicts PC bottleneck percentage using ML"
)

interface.launch()