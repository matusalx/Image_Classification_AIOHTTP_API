import os
from aiohttp import web
import aiohttp_jinja2
import jinja2

from PIL import Image
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms

import json
from dicttoxml import dicttoxml

base_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vgg19_bn_model.pt')
current_dir = os.path.dirname(__file__)

model = models.vgg19_bn()
num_classes = 4
n_inputs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load(base_model_dir))
model.eval()


def model_prediction(image):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                                        ])

    tr_img = data_transform(image)
    tr_img = tr_img.unsqueeze(0)
    prediction = model(tr_img)
    prediction = prediction.argmax().item()
    class_map = {'camera': 0, 'laptop': 1, 'mobile': 2, 'tv': 3}
    class_map = dict([(value, key) for key, value in class_map.items()])
    prediction = class_map[prediction]

    return prediction


@aiohttp_jinja2.template('index_json.html')
async def index(request):
    return web.HTTPFound('/json')

@aiohttp_jinja2.template('index_json.html')
async def json_func(request):
    return None

@aiohttp_jinja2.template('index_xml.html')
async def xml(request):
    return None
routes = web.RouteTableDef()
app = web.Application()


async def file_handler_json(request):
    data = await request.post()
    names = []
    d = {}
    for x, y in data.items():
        image = Image.open(y.file)
        predict_class = model_prediction(image)
        d['filename'] = y.filename
        d['class'] = predict_class
        names.append(d)
    json_resp = json.dumps(names)
    return web.json_response(json_resp)


async def file_handler_xml(request):
    data = await request.post()
    names = []
    d = {}
    for x, y in data.items():

        image = Image.open(y.file)
        predict_class = model_prediction(image)
        d['filename'] = y.filename
        d['class'] = predict_class
        names.append(d)
    xml_resp = dicttoxml(names)

    return web.Response(text=str(xml_resp))


app.add_routes([web.get('/', index),
                web.get('/xml', xml),
                web.get('/json', json_func),
                web.post('/file_handler_json', file_handler_json),
                web.post('/file_handler_xml', file_handler_xml),
                ])

aiohttp_jinja2.setup(
                app, loader=jinja2.FileSystemLoader(current_dir)
                    )
web.run_app(app)
