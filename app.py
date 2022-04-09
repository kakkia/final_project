import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K

@st.cache(allow_output_mutation=True)
def load_model():
  model = load_model('/content/my_model2.hdf5')
  model._make_predict_function()
  model.summary()
  return model

def main():
    st.title("Talking Hands")
    st.header("The first Italian Gestures Classification app")
    from PIL import Image
    image = Image.open('img_streamlit/gestures.jpg')

    st.image(image, use_column_width=True)

    st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. My app will help you to use the appropriate gesture when you just can't find the words.
""")
    
if __name__ == '__main__':
    main()
    uploaded_file = st.file_uploader(label='Give us a hand and upload your picture')
    #sentence = st.text_input('Input your sentence here:')
    model= load_model()
    if uploaded_file:
        y_hat = model.predict(uploaded_file)

#if st.button('Predict your gesture'):
#     st.write('Not just yet. I will upload my model later')

import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(uploaded_file, model):
  size = (180,180)    
  image = ImageOps.fit(uploaded_file, size, Image.ANTIALIAS)
  image = np.asarray(image)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.   
  img_reshape = img[np.newaxis,...]
    
  prediction = model.predict(img_reshape)  
  return prediction

if uploaded_file is None:
    st.text("Please upload an image")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(prediction)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
