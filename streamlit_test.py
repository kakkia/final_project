import streamlit as st

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

def main():
    st.title("Talking Hands")
    st.header("The first Italian Gestures Classification app")
    from PIL import Image
    image = Image.open('img_streamlit/gestures.jpg')

    st.image(image, use_column_width=True)

    st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. My app will help you to use the appropriate gesture when you just can't find the words.\n
I have been training a neural network model to predict three key expressions: "what?", "shoo", and "excellent".
#### How about giving us a hand to test it?""")
    load_image()

if __name__ == '__main__':
    main()


