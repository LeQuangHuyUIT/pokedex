import base64
from PIL import Image
from io import BytesIO
import os
import streamlit as st
from keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px


@st.cache_resource ()
def load_classifier(model_name):
    return load_model(model_name)


@st.cache_data ()
def read_list_name(file_name):
    res = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res.append(line.replace("\n", ""))
    return res


def get_sample_image_path(indices, pokemon_name):
    path = "SampleImages"
    return [pokemon_name[idx] + ".jpg" for idx in indices]

def get_thumbnail(path: str) -> Image:
    img = Image.open(os.path.join("SampleImages", path))
    img = img.resize((256, 256))
    img.thumbnail((256, 256))
    return img

def image_to_base64(img_path: str) -> str:
    img = get_thumbnail(img_path)
    with BytesIO() as buffer:
        img.save(buffer, 'png') # or 'jpeg'
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(img_path: str) -> str:
    return f'<img src="data:image/jpg;base64,{image_to_base64(img_path)}">'


def display_basic_info(match, image, idx):
    # get basic info data
    name = match['name']
    id = match['pokedex_number']
    height = str(match['height_m'])
    weight = str(match['weight_kg'])
    species = ' '.join(match['species'].split(' ')[:-1])
    type1 = match['type_1']
    type2 = match['type_2']
    type_number = match['type_number']
    ability1 = match['ability_1']
    ability2 = match['ability_2']
    ability_hidden = match['ability_hidden']

    col1, col2, col3 = st.columns(3)

    # leftmost column col1 displays pokemon image
    try:

        col1.image(image)
    except:  # output 'Image not available' instead of crashing the program when image not found
        col1.write('Image not available.')

    # middle column col2 displays nicely formatted Pokemon type using css loaded earlier
    with col2.container():
        col2.write('Type')
        # html code that loads the class defined in css, each Pokemon type has a different style color
        type_text = f'<span class="icon type-{type1.lower()}">{type1}</span>'
        if type_number == 2:
            type_text += f' <span class="icon type-{type2.lower()}">{type2}</span>'
        # markdown displays html code directly
        col2.markdown(type_text, unsafe_allow_html=True)
        col2.metric("Height", height + " m")
        col2.metric("Weight", weight + " kg")

    # rightmost column col3 displays Pokemon abilities
    with col3.container():
        col3.metric("Species", species)
        col3.write('Abilities')
        if str(ability1) != 'nan':
            col3.subheader(ability1)
        if str(ability2) != 'nan':
            col3.subheader(ability2)
        if str(ability_hidden) != 'nan':
            col3.subheader(ability_hidden + ' (Hidden)')


def display_base_stats_type_defenses(match):
    # list to gather all type weaknesses and resistances
    weakness_2_types = []
    weakness_4_types = []
    resistance_half_types = []
    resistance_quarter_types = []

    # dataset only shows damage (x4, x2, x0.25, x0.5) of each type towards the Pokemon
    # manually classify the damages into weaknesses and resistances list
    for i, j in match.iterrows():
        for column, value in j.iteritems():
            if column.startswith('against_'):
                type = column.split('_')[1]
                if value == 0.5:
                    resistance_half_types.append(type)
                elif value == 0.25:
                    resistance_quarter_types.append(type)
                elif value == 2:
                    weakness_2_types.append(type)
                elif value == 4:
                    weakness_4_types.append(type)

    with st.container():
        col1, col2 = st.columns(2)

        # left column col1 displays horizontal bar chart of base stats
        col1.subheader('Base Stats')
        # get base stats of Pokemon and rename columns nicely
        df_stats = match[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
        df_stats = df_stats.rename(
            columns={'hp': 'HP', 'attack': 'Attack', 'defense': 'Defense', 'sp_attack': 'Special Attack',
                     'sp_defense': 'Special Defense', 'speed': 'Speed'}).T
        df_stats.columns = ['stats']

        # plot horizontal bar chart using matplotlib.pyplot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(y=df_stats.index, width=df_stats.stats)
        plt.xlim([0, 250])
        col1.pyplot(fig)

        # right column col2 displays the weaknesses and resistances
        # the displayed types are nicely formatted using css (same as earlier)
        col2.subheader('Type Defenses')
        col2.write('Strong Weaknesses (x4)')
        weakness_text = ''
        for type in weakness_4_types:
            weakness_text += f' <span class="icon type-{type}">{type}</span>'
        col2.markdown(weakness_text, unsafe_allow_html=True)
        col2.write('Weaknesses (x2)')
        weakness_text = ''
        for type in weakness_2_types:
            weakness_text += f' <span class="icon type-{type}">{type}</span>'
        col2.markdown(weakness_text, unsafe_allow_html=True)

        col2.write('Resistances (x0.5)')
        resistance_half_text = ''
        for type in resistance_half_types:
            resistance_half_text += f' <span class="icon type-{type}">{type}</span>'
        col2.markdown(resistance_half_text, unsafe_allow_html=True)

        col2.write('Strong Resistances (x0.25)')
        resistance_quarter_text = ''
        for type in resistance_quarter_types:
            resistance_quarter_text += f' <span class="icon type-{type}">{type}</span>'
        col2.markdown(resistance_quarter_text, unsafe_allow_html=True)


def display_radar_chart(match):

    # get base stats of Pokemon and rename columns nicely
    df_stats = match[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
    df_stats = df_stats.rename(
        columns={'hp': 'HP', 'attack': 'Attack', 'defense': 'Defense', 'sp_attack': 'Special Attack',
                 'sp_defense': 'Special Defense', 'speed': 'Speed'}).T
    df_stats.columns = ['stats']

    # use plotly express to plot out radar char of stats
    fig = px.line_polar(df_stats, r='stats', theta=df_stats.index, line_close=True, range_r=[0, 250])
    st.plotly_chart(fig)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)