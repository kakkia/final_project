import streamlit as st
import tensorflow as tf
import keras
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Desktop/WBS_bootcamp/final_project/my_model/my_new_model.hdf5')
    model.summary()
    return model

def main():
    st.title("Talking Hands")
    st.header("The first Italian Gestures Classification app")
    from PIL import Image
    image = Image.open('Desktop/WBS_bootcamp/final_project/img_streamlit/gestures.jpg')

    st.image(image, use_column_width=True)

    st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. My app will help you to use the appropriate gesture when you just can't find the words.
""")
    
if __name__ == '__main__':
    main()
    
uploaded_file = st.file_uploader(label='Give us a hand and upload your picture', type=None)
predict_hand = st.button('Predict your gesture')

############################################### -- so far so good!
          
class_names = ["what", "shoo", "perfect"] 

def predict_class(image):
    classify_model = tf.keras.models.load_model('Desktop/WBS_bootcamp/final_project/my_model/my_new_model.hdf5')
    test_image = image.resize((180,180))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    prediction = classify_model.predict(test_image)
    output = np.argmax(prediction)
    all_preds = prediction[0]
    predicted_class = f"Your hand is predicted as {class_names[output]}"
    return predicted_class, output

def hand_description(output):
    
    if output == "what":
        text = st.markdown("""<h4>What?</h4>
        <ul>
        <li>The tips of the fingers of one hand are brought sharply together to form an upward-pointing cone.</li>
        <li>The hand can either be held motionless or be shaken more or less violently up and down.</li>
        <li>How fast you move it, depends on the degree of impatience expressed.</li>
        <li>Don't be afraid of using it when someone tells you something unexpectedly upsetting.</li>
        </ul>""", True)
    elif output == "shoo":
        text = st.markdown("""<h4>Shoo</h4>
        <ul>
        <li>The flat hand slowly moves as to follow the people you are addressing.</li>
        <li></li>
        <li>In parenting, it can be moved up and down to suggest you will be punished for what you did.</li>
        <li></li>
        </ul>""",True)
    else:
        text = st.markdown("""<h4>Excellent</h4>
        <ul>
        <li>This gesture express both approval and hearty satifaction.</li>
        <li>It is typical of the good-natured and contented gourmet.</li>
        <li>Use it anythime you find something delicious, or you completely agree with someone.</li>
        <li>And it's not just for food.</li>
        </ul>""",True)
    return text

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your hand', width=None)
else:
    st.warning('After uploading your picture click: Predict your gesture')

if predict_hand:   
    result, output = predict_class(image)
   # st.subheader(result)
    st.text = hand_description(output)
else:
    st.write('')
