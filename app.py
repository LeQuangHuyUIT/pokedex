import os

import tensorflow as tf
import streamlit as st
from streamlit_cropper import st_cropper
from utils import *
import pandas as pd
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)
pd.set_option('display.colheader_justify', 'center')
IMG_SIZE = 256
def get_predict_result(input_image, pokemon_name):
    tensor = tf.keras.utils.img_to_array(input_image) / 255.
    tensor = tf.expand_dims(tensor, 0)
    # st.write(tensor.shape)
    if tensor.shape != (1, IMG_SIZE, IMG_SIZE, 3):
        if tensor.shape[-1] > 3:
            tensor = tensor[:, :, :, :3]
        else:
            temp = tensor[:, :, :, :1]
            tensor = tf.concat([temp, temp, temp], axis= -1)
        # tensor = tf.reshape(tensor, [1, 256, 256, 3])
        # print(tensor.shape)
    y_pred = model.predict(tensor)

    result = tf.math.top_k(y_pred[0], k=1)
    indices = list(result.indices.numpy())
    names = [pokemon_name[idx] for idx in indices]
    values = [f'{(val * 100):.2f}%' for val in list(result.values.numpy())]
    images = get_sample_image_path(indices, pokemon_name)

    _dict = {'Image': images, "name": names, "confidence": values}
    df = pd.DataFrame(_dict)
    return df


# Upload an image and set some options for demo purposes
st.set_page_config(layout="wide")
st.header("Pokemon Classification ")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
use_cropper = st.sidebar.checkbox(label="Use Image cropper", value=False)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
press_predict = st.sidebar.button("Predict")
aspect_ratio = None


########################################################################
with st.spinner('Loading model...'):
    model = load_classifier("weights/Xception-0.1410-weights-10.hdf5")

pokemon_name = read_list_name("pokemon_name.txt")
database = pd.read_csv('Pokemon.csv', encoding= 'unicode_escape')
local_css('style.css')


if img_file:
    img = Image.open(img_file)
    img = img.resize((256, 256))

    if use_cropper:
        # Get a cropped image from the frontend
        input_image = st_cropper(img, realtime_update=True, box_color=box_color,
                                 aspect_ratio=aspect_ratio)

        # Manipulate cropped image at will
        st.write("Preview")
        _ = input_image.thumbnail((256, 256))
        st.image(input_image)
        input_image = input_image.resize((IMG_SIZE, IMG_SIZE))
        st.image(input_image)
    else:
        st.image(img)
        input_image = img.resize((IMG_SIZE, IMG_SIZE))

    if press_predict:
        with st.spinner('Predict'):
            df = get_predict_result(input_image, pokemon_name)
            # st.markdown(df.to_html(escape=False, formatters=dict(Image=image_formatter), index=False), unsafe_allow_html=True)

            pokemon_name = df['name'].tolist()[0]
            # pokemon_name = pokemon_name.lower()
            relevant_df = database[database['name'].str.contains(pokemon_name)]
            relevant_name = relevant_df['name'].tolist()
            # st.markdown(relevant_df.to_html(), unsafe_allow_html=True)


            for i, tab in enumerate(st.tabs(relevant_name)):
                with tab:
                    st.header(relevant_name[i])
                    image_path = os.path.join('pokemon_images', relevant_name[i] + '.jpg')
                    image = Image.open(image_path)

                    display_basic_info(relevant_df.iloc[i], image, i)
                    display_base_stats_type_defenses(relevant_df[relevant_df['name'] == relevant_name[i]])
                    display_radar_chart(relevant_df[relevant_df['name'] == relevant_name[i]])
