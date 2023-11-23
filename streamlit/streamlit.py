import streamlit as st
import pandas as pd
import numpy as np
import json
from moviepy.editor import ImageSequenceClip

league1_key = 'select_league1'
league2_key = 'select_league2'
team1_key = 'select_team1'
team2_key = 'select_team2'


# def show_interpolation(saltos, team1, team2):

df = pd.read_excel('../logos-teams.xlsx')

st.title('Escudemosnos')

st.image('../bover.png', use_column_width=True)

left_column, right_column = st.columns(2)

with left_column:
    league1 = st.selectbox('Elija una liga', df['league'].unique(), key=league1_key)
    team1_options = df[df['league'] == league1]['name']
    team1 = st.selectbox('Elija el primer equipo', team1_options, key=team1_key)
    

with right_column:
    league2 = st.selectbox('Elija una liga', df['league'].unique(), key=league2_key)
    team2_options = df[df['league'] == league2]['name']
    team2 = st.selectbox('Elija el segundo equipo', team2_options, key=team2_key)

saltos = st.select_slider('Elija la cantidad de saltos',range(3,8), value=5)

col1, col2, col3 = st.columns(3)

if st.button('Mostrar Interpolación'):
        show_interpolation(saltos,team1, team2)

with col1:
    path = df[df["name"] == team1]["img_dir"]
    st.image(path.iloc[0], use_column_width=True)

with col3:
    path = df[df["name"] == team2]["img_dir"]
    st.image(path.iloc[0], use_column_width=True)



