import streamlit as st

st.title("Talking Hands")

st.header("The first Italian Gestures Classification app")

from PIL import Image
image = Image.open('https://github.com/kakkia/final_project/blob/main/img_streamlit/gestures.jpg')

st.image(image, use_column_width=True)

st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. This app helps you to use the appropriate gesture when you just can't find the words.\n
So far I have trained a model to predict three key expressions: "what?", "shoo", and "excellent".
How about giving us a hand to test it?""")

uploaded_file = st.file_uploader("Upload a picture to predict your gesture", type="jpg, png")


