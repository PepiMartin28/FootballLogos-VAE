from pydantic import BaseModel
from fastapi import FastAPI, Response
import json
import torch
from modelo.model_definition import Variational_Autoencoder
from modelo.functions import model_interp, model_centroid

VERSION = "0.0.1"

print('Cargando aplicación...')
app = FastAPI()

print(f'Cargando modelo con versión: {VERSION}')
vae = Variational_Autoencoder(700)
vae.load_state_dict(torch.load(f'vae-weights-700DV6.params', map_location="cpu"))
vae.eval()
print('Modelo cargado')

class InterpolationRequest(BaseModel):
    League1: str
    Team1: str
    League2: str 
    Team2: str
    size: int = 100

@app.post("/interpolation")
def interpolation(request_body: InterpolationRequest, response: Response):

    interp_result = model_interp(
        model = vae,
        league1 = request_body.League1,
        team1 = request_body.Team1,
        league2 = request_body.League2, 
        team2 = request_body.Team2, 
        size = request_body.size)

    interp_result = interp_result.permute(0,2,3,1)

    return {
        "images": json.dumps(interp_result.tolist())
    }

class CentroidRequest(BaseModel):
    League: str

@app.post("/centroid")
def centroid(request_body: CentroidRequest, response: Response):

    centroid_result = model_centroid(
        model = vae,
        league = request_body.League
    )

    centroid_result_as_list = [(t.tolist() for t in tupla) for tupla in centroid_result]

    return {
        "images": centroid_result_as_list
    }