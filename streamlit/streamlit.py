import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from moviepy.editor import ImageSequenceClip

league1_key = 'select_league1'
league2_key = 'select_league2'
team1_key = 'select_team1'
team2_key = 'select_team2'
league_key = 'select_league'

def show_interpolation(team1, league1, team2, league2, size=100):
    data = {
        'Team1': str(team1),
        'League1':str(league1),
        'Team2':str(team2),
        'League2':str(league2),
        'size':int(size)
    }

    result = requests.post(f"http://127.0.0.1:8000/interpolation", json = data)
	
    if result.status_code != 200:
        st.error(f'There was an error in backend')

    else:
        result = result.json()
        data = np.asarray(json.loads(result["images"]))
        print(data.shape)
        clip = ImageSequenceClip(list(data * 255), fps=20)
        clip.write_gif('local.gif', fps=20)
        with col2:
            st.image("local.gif", use_column_width=True)

def show_centroids(league):
    if len(df[df['league'] == league]) < 30:
        st.error('La liga debe tener mas de 30 escudos')

    data={
        'League':str(league)
    }
    result = requests.post(f"http://127.0.0.1:8000/centroid", json = data)
	
    if result.status_code != 200:
        st.error(f'There was an error in backend')

    else:
        result = result.json()
        data = np.asarray(result['images'])
        for tuple in data:
            with cen1:
                st.image(tuple[0], use_column_width=True)
            with cen2:
                st.image(tuple[1], use_column_width=True)

df = pd.read_excel('../logos-teams.xlsx')

st.title('Escudemosnos')

st.image('../bover.png', use_column_width=True)

left_column, center_column, right_column = st.columns(3)

with left_column:
    league1 = st.selectbox('Elija una liga', df['league'].unique(), key=league1_key)
    team1_options = df[df['league'] == league1]['name']
    team1 = st.selectbox('Elija el primer equipo', team1_options, key=team1_key)
    
with center_column:
    interpolation = st.button('Mostrar Interpolación',use_container_width=True)

with right_column:
    league2 = st.selectbox('Elija una liga', df['league'].unique(), key=league2_key)
    team2_options = df[df['league'] == league2]['name']
    team2 = st.selectbox('Elija el segundo equipo', team2_options, key=team2_key)

col1, col2, col3 = st.columns(3)

with col1:
    path = df[(df["name"] == team1) & (df["league"] == league1)]["img_dir"]
    st.image(path.iloc[0], use_column_width=True)

with col3:
    path = df[(df["name"] == team2) & (df["league"] == league2)]["img_dir"]
    st.image(path.iloc[0], use_column_width=True)

if interpolation:
    show_interpolation(team1, league1, team2, league2)

st.title('Centroides')

league = st.selectbox('Elija una liga', df['league'].unique(), key=league_key)

centroid = st.button('Mostrar centroides')

cen1, cen2 = st.columns(2)

if centroid:
    show_centroids(league)
