from generate_post import generate_post
import gradio as gr

gr.Interface(
    fn=generate_post,
    inputs=gr.Textbox(label="Enter a topic (e.g. AI, career, leadership)"),
    outputs=gr.Textbox(label="Generated LinkedIn Post"),
    title="LinkedIn Post Generator"
).launch()