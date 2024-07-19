import json
import glob
import os 
import csv
import pandas as pd
import torch
import torch.nn as nn
import sys
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import defaultdict
from PIL import Image
import openai
import re

def caption(img_path):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6"

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16
    )
    model.to(device)
    
    img_folder = img_path
    img_files = [f for f in os.listdir(img_folder) if f.endswith('jpg')]

    for img_file in img_files:
        image_path = os.path.join(img_folder, img_file)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs)
        caption = processor.batch_decode(output_ids, skip_spectial_tokens=True)[0]
        caption = re.sub(r'</s>','',caption)
        #print(caption)

        openai.api_key ="*********************************"
        gpt_model = "gpt-3.5-turbo"
        EXTRACT_PROMPT = "Please extract the words related to Architecture, People, Food&Drink, Dance&Music, and Religion from caption. if not have related word, please say N/A"
        PROMPT = EXTRACT_PROMPT + "\n"+\
                "Caption: " + caption
        #print(PROMPT)
        messages = []
        messages = [{"role": "user", "content": PROMPT}]
        response = openai.ChatCompletion.create(model=gpt_model, messages=messages, temperature=0.6)
        answer = response['choices'][0]['message']['content']
        #print(answer)

        INSTRUCT_PROMPT = "I give you the QA results, please change the Caption based on the QA results."\
                        "Do not simply attach the QA result to a caption when changing the caption."\
                        "Don't add additional information. Use all QA results. make one sentence."
        PROMPT = INSTRUCT_PROMPT+"\n\n"+\
                "Caption" + caption + "\n\n"+\
                "QA reslut: "
        culture_question = pd.read_csv('Question_prompt.csv')
        lines = answer.strip().split('\n')
        
        for line in lines:
            category, answer = line.split(': ',1)
            if answer != 'N/A':
                print(category)
                if category == 'People':
                    question = culture_question[culture_question['category'].str.lower()=='clothing']
                    #print(question['prompt'].values[0])
                    question = question['prompt'].values[0]
                else:
                    question = culture_question[culture_question['category'].str.lower()==category.lower()]
                    #print(culture_question['prompt'].values[0])
                    question = question['prompt'].values[0]
                question_prompt = f"Question: {question} Answer:"
                inputs = processor(image, question_prompt, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=51)
                generated_text = processor.batch_decode(generated_ids, skip_spectial_tokens=True)[0].strip()
                generated_text = re.sub(r'</s>','',generated_text)
                QA = "{"+f"Question: {question} Answer: {generated_text}" + "},"
                PROMPT = PROMPT + QA
        #print(PROMPT) 
    
        messages = []
        messages = [{"role":"user","content": PROMPT}]
        response = openai.ChatCompletion.create(model=gpt_model, messages=messages, temperature=0.6)
        answer = response['choices'][0]['message']['content']
        print(answer)
    
    return