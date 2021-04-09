import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import cv2
from io import BytesIO
import base64
import numpy as np
import skimage.exposure
import PIL.Image as Image

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Skin Lesion Photo Classifier'
server = app.server


device = torch.device('cpu')
model = torch.load('model_conv_resnet50.pth', map_location=device)
labels = np.array(open("class.txt").read().splitlines())

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app.layout = html.Div([
    html.Div([
        html.H2('Skin Lesion Photo Classifier'),
        html.Strong('This application detects whether an image needs to be retaken because it is too blurry and/or low contrast.', style={'fontSize': 18}),
        html.Br(),
        html.Strong('If the image is good enough, it will tell you whether to schedule the patient for a consultation.', style={'fontSize': 18}) 
    ]),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):
    #convert uploaded image file in Pillow image file
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image_pil = Image.open(bytes_image).convert('RGB')
    image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), 1)

    blur_threshold = 10
    contrast_threshold = 0.05

    blur_measure = cv2.Laplacian(image, cv2.CV_64F).var()
    if blur_measure > blur_threshold:
        blur_message = f"This image is clear. The Laplacian variance is {blur_measure:.2f}, which is greater than the threshold of {blur_threshold:.2f}. Higher is better."
    else:
        blur_message = f"This image is too blurry. The Laplacian variance is {blur_measure:.2f}, which is less than the threshold of {blur_threshold:.2f}. Higher is better."

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_mean = hsv[...,2].mean()
    

    low_contrast = False
    threshold = 0
    while not low_contrast:
        threshold = round(threshold + 0.02, 3)
        low_contrast = skimage.exposure.is_low_contrast(image, fraction_threshold=threshold)

    if threshold > contrast_threshold:
        contrast_message = f"This image has good contrast. The Gamma is {threshold:.2f}, which is greater than the threshold of {contrast_threshold:.2f}. Gamma range from 0 to 1, and higher is better."
    else:
        contrast_message = f"This image has low contrast. The Gamma is {threshold:.2f}, which is less than the threshold of {contrast_threshold:.2f}. Gamma range from 0 to 1, and higher is better."


    img = preprocess(image_pil)
    img = img.unsqueeze(0)
    pred = model(img)
    #print(pred.detach().numpy())
    prediction = labels[torch.argmax(pred)]
    prob = F.softmax(pred, dim=1)
    df = pd.DataFrame({'Class':labels, 'Probability':prob[0].detach().numpy()*100})
    output_table = generate_table(df.sort_values(['Probability'], ascending=[False]))

    if blur_measure > blur_threshold and threshold > contrast_threshold:
        if prediction == labels[1] :
            output_text = html.H3(f"Please schedule patient for a consultation.", style={'color': 'blue', 'font-weight' : 'bold' })
        elif prediction == labels[0]:
            output_text = html.H3(f"Patient does not need another appointment.", style={'color': 'green', 'font-weight' : 'bold' })
        else:
            output_text = html.H3(f"Something went wrong")


    else:
        output_text = html.H4(f"Please retake your image", style={'color': 'red', 'font-weight' : 'bold' })


    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        output_text,
        html.Strong(blur_message),
        html.Br(),
        html.Strong(contrast_message),
        #html.Br(),
        #html.Strong(f"The brightness of this image is {brightness_mean}"),
        html.Br(),
        html.Img(src=contents, style={"max-height" : "500px" }),
        html.Plaintext(filename),
        html.Plaintext(datetime.datetime.now().strftime("%c")),
        html.Hr(),
        output_table
    ])


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                #html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                html.Td(dataframe.iloc[i][0]), html.Td("{:.2f}%".format(dataframe.iloc[i][1]), style={'text-align' : 'right'}) 
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    #app.run_server(debug=True, port=8050)
    app.run_server(host='0.0.0.0',debug=True, port=8050)
