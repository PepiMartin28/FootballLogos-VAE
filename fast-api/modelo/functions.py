import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.cluster import KMeans

df = pd.read_excel('../logos-teams.xlsx')

transformaciones = transforms.Compose([transforms.Resize((200, 200)),transforms.ToTensor()])

def get_interp(v1, v2, n):
    if not v1.shape == v2.shape:
        raise Exception('Different vector size')

    return np.array([np.linspace(v1[i], v2[i], n) for i in range(v1.shape[0])]).T


def model_interp(model, league1, team1, league2, team2, size = 10):

    image1 = df[(df['name']==team1)&(df['league']==league1)]['img_dir']
    image1 = transformaciones(Image.open(f'{image1.iloc[0]}').convert("RGBA"))

    image2 = df[(df['name']==team2)&(df['league']==league2)]['img_dir']
    image2 = transformaciones(Image.open(f'{image2.iloc[0]}').convert("RGBA"))

    img1_compressed,_,_ = model.encoder(image1.unsqueeze(0))
    img2_compressed,_,_ = model.encoder(image2.unsqueeze(0))

    interp = get_interp(img1_compressed.detach().cpu(), img2_compressed.detach().cpu(), size)

    interp = torch.from_numpy(interp)

    interp = interp.permute(1,0,2).unsqueeze(3)

    artificial_images = model.decoder(interp)

    return artificial_images

def get_centroids(model,images):
    kmeans = KMeans(
        init='k-means++',
        n_clusters=10,
        n_init=50,
        max_iter=500,
        random_state=42
    )
    
    kmeans.fit(images.detach())

    centroids = kmeans.cluster_centers_
    
    labels = kmeans.labels_
    
    imagenes = []
    for cluster_id in range(kmeans.n_clusters):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]

        input = torch.stack([torch.tensor(centroids[cluster_id]), images[indices[0]]]).to(torch.float32)

        imgs = model.decoder(input.unsqueeze(2).unsqueeze(3)).detach()

        imagenes.append([img.permute(1,2,0).detach() for img in imgs])
    
    return imagenes

def model_centroid(model, league):
    data = df[df['league']==league]['img_dir']

    data_images = torch.stack([transformaciones(Image.open(image).convert("RGBA")) for image in data])

    latent_images,_,_ = model.encoder(data_images)

    artificial_images = get_centroids(model, latent_images)
    
    return artificial_images

