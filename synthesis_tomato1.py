# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:22:20 2022

@author: Payam_(cyrus)
"""



import PIL
from PIL import Image
from random import randint
import pandas as pd
import requests
from io import BytesIO
import zipfile

OFFSET = 1 
COUNT  = 10

print("Generating tomatos Images")

# Load the paperclip and background images
response_tomatoes = requests.get("https://raw.githubusercontent.com/PayamCyrus/tomatoes_count/main/1-fococlipping-standard_payam.png")
tomatoes = Image.open(BytesIO(response_tomatoes.content)).convert("RGBA")

response_background = requests.get("https://raw.githubusercontent.com/PayamCyrus/tomatoes_count/main/2.png")
background = Image.open(BytesIO(response_background.content)).convert("RGBA")

# Resize the background
background = background.resize((128,128),resample=PIL.Image.LANCZOS)

# Keep the apsect ratio of the paperclips when they are scaled
aspect = tomatoes.width/tomatoes.height
tomatoes_count = []

# We will output to a ZIP file that can be easily downloaded
z = zipfile.ZipFile('payam dataset tomatoes5.zip', 'w', zipfile.ZIP_DEFLATED)

for c in range(COUNT):
    render_img = background.copy()
    cnt = randint(0,100)
    tomatoes_count.append(cnt)
    for i in range(cnt):
        a = randint(0,115)
        tomatoes_size = randint(3,15)
        w = int(tomatoes_size*aspect)

        tomatoes2 = tomatoes.resize((w,tomatoes_size),resample=PIL.Image.LANCZOS)


        x = randint(-int(tomatoes2.width/2),background.width - int(tomatoes2.width/2))
        y = randint(-int(tomatoes2.height/2),background.height - int(tomatoes2.height/2))

        tomatoes2 = PIL.Image.Image.rotate(tomatoes2,a,resample=PIL.Image.BICUBIC,expand=True)


        render_img.paste(tomatoes2,(x,y),tomatoes2)

    image_file = BytesIO()
    render_img.save(image_file, 'PNG')
    z.writestr(f'tomato-{c+OFFSET}.png', image_file.getvalue())
    
df = pd.DataFrame( {'id':range(OFFSET,len(tomatoes_count)+OFFSET),'tomatoes_count':tomatoes_count})
z.writestr('payam.csv', df.to_csv(index=False))
z.close()
print("done")

df

