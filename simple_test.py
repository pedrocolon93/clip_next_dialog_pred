import datetime
import os
import re

import torch
from torch.nn import LSTM
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartForConditionalGeneration, BartTokenizer

import clip
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
folder = "data2/"
items = sorted(list(filter(lambda x: ".mp4" in x, os.listdir("data2"))), reverse=True)
window_average_size = 1  # average for the window size
window_size = 11
gpt_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
gpt_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# gpt_model.resize_token_embeddings(len(gpt_tokenizer))

epochs = 2
training_items, testing_items = train_test_split(items)
max_dur = 0
fps = 30
afps = 25


class WindowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window_encoder = LSTM(input_size=256, num_layers=3, bidirectional=True, batch_first=True, hidden_size=512)
        self.window_decoder = LSTM(input_size=1024, num_layers=3, bidirectional=True, hidden_size=256)
        self.window_compressor = torch.nn.Linear(1071, 512)
        self.window_compressor_2 = torch.nn.Linear(512, 256)
        self.embeddings = torch.nn.Embedding(num_embeddings=5, embedding_dim=256)
        self.project_up = torch.nn.Linear(512, 768)
        self.max_size = 16

    def forward(self, x):
        x = x.float()
        h = self.window_compressor(x)
        h = self.window_compressor_2(h)
        h = torch.cat([self.embeddings(torch.tensor([0])), h, self.embeddings(torch.tensor([1]))])
        h = torch.cat([h] + [self.embeddings(torch.tensor([2]))] * (self.max_size - h.shape[0]))
        h = h.unsqueeze(0)
        h_e = self.window_encoder(h)[0]
        h_d = self.window_decoder(h_e)[0]
        h_d = self.project_up(h_d)
        return h_e, h_d


window_model = WindowModel()
optimizer = Adam(params=list(window_model.parameters())+list(gpt_model.parameters()))
optimizer.zero_grad()
for item in items:
    # try:
    affdex = pd.read_csv(folder + item.replace(".mp4", "") + "_affdex.csv")
    transcript = pd.read_csv(folder + item.replace(".mp4", "") + "_rev.csv")
    transcript["content"] = transcript["content"].apply(lambda x: re.sub('\[.*\]', "", x))
    transcript["content_plus_speaker"] = transcript["speaker"] + ": " + transcript["content"]
    pair1_idxs = [_ for _ in range(transcript.shape[0])]
    pair2_idxs = [_ for _ in range(1, transcript.shape[0])] + [None]
    if len(pair1_idxs) == 1:
        print("Only one interaction.. continuing")
        continue
    start = affdex.iloc[0]["TimeStamp"]  # time in seconds where the interaction starts
    for pair in tqdm(zip(pair1_idxs, pair2_idxs)):
        if pair[1] is None or pair[0] is None:
            print("None pairs",pair)
            continue
        next_text = transcript.iloc[pair[1]]
        prior_text = transcript.iloc[pair[0]]

        to_time_s = transcript.iloc[pair[1]]['timestamp']
        to_time_s = datetime.datetime.strptime(to_time_s, "%H:%M:%S") - datetime.datetime(1900, 1, 1)
        to_time_s = to_time_s.total_seconds()
        from_time_s = transcript.iloc[pair[0]]['timestamp']
        from_time_s = datetime.datetime.strptime(from_time_s, "%H:%M:%S") - datetime.datetime(1900, 1, 1)
        from_time_s = from_time_s.total_seconds()
        absolute_start_time_s = from_time_s - start
        absolute_end_time_s = to_time_s - start
        if absolute_start_time_s < 0:
            absolute_start_time_s = 0.0
        total_duration = absolute_end_time_s - absolute_start_time_s
        max_dur = max(total_duration, max_dur)
        # Get the video frames for the interaction
        video_frames = []
        # print(pair)
        cap = cv.VideoCapture(folder + item)
        video_loaded = False
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print(folder+item)
                print("Can't receive frame (stream end?). Exiting ...")
                break
            video_loaded = True
            ts = cap.get(cv.CAP_PROP_POS_MSEC) / 1000
            if absolute_start_time_s <= ts and ts <= absolute_end_time_s:
                video_frames.append(frame)
            # cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        if len(video_frames) == 0 or not video_loaded:
            print("Could not load video")
            continue
        cap.release()
        cv.destroyAllWindows()
        # Affdex for the current frames
        affdex_frames = affdex[(from_time_s <= affdex["TimeStamp"]) & (affdex["TimeStamp"] <= to_time_s)]
        print("here")
        print("Max duration", max_dur)
        video_sequence = []
        affdex_sequence = []
        ts = 0
        last_time = 0

        for i in range(1, window_size + 1):
            vfs = (i - 1) * fps * window_average_size
            vff = (i) * fps * window_average_size
            img_avg = np.average(video_frames[vfs:vff], axis=0)
            # pil_images = Image.[Image.fromarray(img_avg[:,:,x]) for x in range(3)]
            if len(img_avg.shape) == 0:
                break
            # print(img_avg.shape)
            Img_avg = Image.fromarray(img_avg, mode="RGB")
            video_sequence.append(Img_avg)
            affdex_offset = affdex_frames.iloc[0]["TimeStamp"]
            afs = affdex_offset + (i - 1) * window_average_size
            aff = affdex_offset + (i) * window_average_size
            fs = affdex_frames[(affdex_frames["TimeStamp"] >= afs) & (affdex_frames["TimeStamp"] < aff)]
            fs = fs.drop(['TimeStamp', 'TimeStamp_td', 'x', 'y',
                          'width', 'height'], axis=1)
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            fs = fs.select_dtypes(include=numerics)
            # fs = fs.drop(index=0,axis=1)
            fs = fs.dropna(axis=1)
            # fs = fs.drop(['TimeStamp','age','TimeStamp_td'],axis=1)
            fsnp = fs.to_numpy()[:, 1:]
            if fsnp.shape[1]==0:
                break
            # print(fsnp.shape,len(video_sequence),len(affdex_sequence))
            affdex_sequence.append(np.expand_dims(np.average(fsnp, axis=0), axis=0))
        if len(video_sequence)!=len(affdex_sequence):
            continue
        # AFFDEX_SEQUENCE = 47
        # Video_encoding = 512
        # Text_encoding = 512
        affdex_sequence = torch.cat([torch.from_numpy(x).to(device) for x in affdex_sequence], dim=0)
        video_encoding_sequence = torch.cat(
            [model.encode_image(preprocess(x).to(device).unsqueeze(0)) for x in video_sequence], dim=0)
        text_encoding_sequence = torch.cat(
            [model.encode_text(clip.tokenize(prior_text["content_plus_speaker"], truncate=True).to(device)) for _ in
             range(len(video_sequence))], dim=0)
        all = torch.cat([affdex_sequence, video_encoding_sequence, text_encoding_sequence], dim=1)
        encoding = window_model(all)
        labels = None
        prior_text_tok = gpt_tokenizer(prior_text["content_plus_speaker"],
                                       padding="max_length",max_length=384,
                                       add_special_tokens=True,
                                       return_tensors='pt')
        prior_text_embeds = gpt_model.model.shared(prior_text_tok["input_ids"])
        a= prior_text_embeds[:,0, :].unsqueeze(0)
        b= encoding[1]
        c= prior_text_embeds[:,1:, :]
        inputs_embeds = torch.cat([a,b,c],dim=1)
        with gpt_tokenizer.as_target_tokenizer():
            labels = gpt_tokenizer(next_text["content_plus_speaker"], max_length=128, truncation=True,
                                   padding="max_length",return_tensors='pt')
            labels["input_ids"] = torch.tensor([
                [(l if l != gpt_tokenizer.pad_token_id else -100) for l in label] for label in
                labels["input_ids"]
            ])

        batch_ipt = {
            "inputs_embeds": encoding[1],
            "labels": labels["input_ids"]
        }
        loss = gpt_model(**batch_ipt)['loss']
        loss.backward()
        print("Loss",loss.item())
        optimizer.step()
        optimizer.zero_grad()
        print("Prediction...")
        p = gpt_model.generate(inputs_embeds=batch_ipt["inputs_embeds"],
                               min_length=5,max_length=128,num_beams=5,
                               num_return_sequences=1).squeeze(0)
        print(gpt_tokenizer.decode(p.tolist()))

    # except Exception as e:
    #     print(e)
    #     print("Skipping",item)
