import gradio as gr
from matplotlib.pyplot import title
import numpy as np
import tensorflow as tf
import random 
from tensorflow import keras
import json 
import requests

def get_rest_url(model_name, host='127.0.0.1', port='8501', verb='predict'):
    url = 'http://{0}:{1}/v1/models/{2}:predict'.format(host, port, model_name)
     
    return url

def rest_request(data, url):
    payload = json.dumps({'instances': data.tolist()})
    response = requests.post(url=url, data=payload)
    return response


damage_types = np.array(sorted(['disaster happened', 'no disaster happened']))
disaster_types = np.array(sorted(['volcano', 'flooding', 'earthquake', 'fire', 'wind', 'tsunami']))
damage_levels = np.array(['no damage', 'minor damage', 'major damage', 'destroyed'])



def damage_classification(img):
    # prediction = np.random.rand(1, 2)[0]
    # return {damage_types[i]: prediction[i] for i in range(len(damage_types))}

    image = np.zeros((1, 1024, 1024, 3), dtype=np.uint8)
    image[0] = img
    results = json.loads(rest_request(image, get_rest_url(model_name='binary-damage-classification-model', host='54.89.217.229')).content)
    prediction = results['predictions'][0]

    return {damage_types[i]: prediction[i] for i in range(len(damage_types))}


def disaster_classification(img):
    image = np.zeros((1, 1024, 1024, 3), dtype=np.uint8)
    image[0] = img
    # prediction = model.predict(image).tolist()[0]
    # prediction = np.random.rand(1, 6)[0]
    results = json.loads(rest_request(image, get_rest_url(model_name='disaster-classification-model', host='3.86.228.238')).content)
    prediction = results['predictions'][0]

    return {disaster_types[i]: prediction[i] for i in range(len(disaster_types))}


def regional_damage_classification(img):
    image = np.zeros((1, 1024, 1024, 3), dtype=np.uint8)
    image[0] = img
    results = json.loads(rest_request(image, get_rest_url(model_name='regional-damage-classification-model', host='54.145.173.193')).content)
    prediction = results['predictions'][0]

    return {damage_levels[i]: prediction[i] for i in range(len(damage_levels))}



iface = gr.Interface(
    fn = [damage_classification, disaster_classification, regional_damage_classification],
    inputs = gr.inputs.Image(shape=(1024, 1024), image_mode='RGB', invert_colors=False, source="upload", type='numpy'), 
    outputs = gr.outputs.Label(),
    allow_screenshot=True, 
    allow_flagging='never',
    examples=[
        './sample_images/hurricane.png',
        './sample_images/volcano.png',
        './sample_images/wildfire.png',
        './sample_images/earthquake.png',
        './sample_images/tsunami.png',
    ],
    title="Soteria - AI for Natural Disaster Response",
    description=""" 
        Check out our project @ https://github.com/Soteria-ai/Soteria for more explantation! Demo below takes ~15 seconds to get the results.
    """,
    theme="grass",
)
iface.launch(share=False, show_error=True, inline=True, debug=True)