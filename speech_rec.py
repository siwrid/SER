import numpy as np
import pandas as pd
import transformers
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
from datasets import Audio, Dataset
import pickle

#### STEP 0: Load the data from the pickle file
data_path = 'IEMOCAP_features.pkl'

with open(data_path, 'rb') as file:
    (
        videoIDs, videoSpeakers, videoLabels, videoText,
        videoAudio, videoVisual, videoSentence, trainVid,
        testVid
    ) = pickle.load(file, encoding='latin1')

"""
trainVid (DICT dialogueId -> UtteranceID List): all dialogue IDs for trainset (total: 120)
testVid (LIST UtteranceID Set): all dialogue IDs for testset (total: 31)
videoIDs (LIST UtteranceID Set): all dialogue IDs for whole dataset (total: 120 + 31)

videoSpeakers : There are multiple participators in one dialogue. videoSpeakers maps utterance to its speakers
videoLabels : The emotion Labels for each utterance in a dialogue.
videoText : The text features extracts using TextCNN.
videoAudio : The video features extracts using openSMILE kitools
videoVisual : The visual features extracts using 3d-CNN.
videoSentence : The raw text info in a dialogue.
"""

# STEP 1 : label and corresponding value 
emotions={
  '0':'happy',
  '1':'sad',
  '2':'neutral',
  '3':'angry',
  '4':'excited',
}

#### STEP 2 : Filter the data (all data with label 5 are not considered because we didn't compute their accuracy with Chat GPT)

filtered_data = {
    'videoIDs': [],
    'videoSpeakers': {},
    'videoLabels': {},
    'videoText': {},
    'videoAudio': {},
    'videoVisual': {},
    'videoSentence': {},
}

for vid in videoIDs:
    for speaker, label, text, audio, visual, sentence in zip(videoSpeakers[vid], videoLabels[vid], videoText[vid], videoAudio[vid], videoVisual[vid], videoSentence[vid]):
        if label != 5:
            if vid not in filtered_data['videoIDs']:
                filtered_data['videoIDs'].append(vid)
                filtered_data['videoSpeakers'][vid] = []
                filtered_data['videoLabels'][vid] = []
                filtered_data['videoText'][vid] = []
                filtered_data['videoAudio'][vid] = []
                filtered_data['videoVisual'][vid] = []
                filtered_data['videoSentence'][vid] = []
            
            filtered_data['videoSpeakers'][vid].append(speaker)
            filtered_data['videoLabels'][vid].append(label)
            filtered_data['videoText'][vid].append(text)
            filtered_data['videoAudio'][vid].append(audio)
            filtered_data['videoVisual'][vid].append(visual)
            filtered_data['videoSentence'][vid].append(sentence)

#### STEP 3: put the filtered data (no label 5) into a dataframe  
data_list = []
for vid in filtered_data['videoIDs']:
    for speaker, label, text, audio, visual, sentence in zip(
            filtered_data['videoSpeakers'][vid],
            filtered_data['videoLabels'][vid],
            filtered_data['videoText'][vid],
            filtered_data['videoAudio'][vid],
            filtered_data['videoVisual'][vid],
            filtered_data['videoSentence'][vid]
    ):
        data_list.append({
            'VideoID': vid,
            'Speaker': speaker,
            'Label': label,
            'Text': text,
            'Audio': audio,
            'Visual': visual,
            'Sentence': sentence
        })

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

# Display the DataFrame
print(df)
