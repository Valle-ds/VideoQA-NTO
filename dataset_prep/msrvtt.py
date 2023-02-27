from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from data.utils import pre_caption

decord.bridge.set_bridge("torch")

class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
    
    
class VideoDataset(Dataset):

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/msrvtt_test.jsonl'
        filename = 'msrvtt_test.jsonl'

        download_url(url,ann_root)
        self.annotation = load_jsonl(os.path.join(ann_root,filename))
        
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.text = [pre_caption(ann['caption'],40) for ann in self.annotation]
        self.txt2video = [i for i in range(len(self.annotation))]
        self.video2txt = self.txt2video               
            
            
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]  

        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt) 

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())
        
        return video, ann['clip_name']

        

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms
    



import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        
        if split=='train':
            urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
            self.annotation = []
            for f in train_files:
                download_url(urls[f],ann_root)
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json',ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,'vqa_test.json'),'r'))    
            
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'])        
            
            if ann['dataset']=='vqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.2]  

            return image, question, answers, weights
        
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n   


import numpy as np
from .video_base_dataset import BaseDataset
import os
import json
import pandas as pd


class MSRVTTQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        #         if split == "test":
        #             split = "val"
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["msrvtt_qa_train"]
            # names = ["msrvtt_qa_train", "msrvtt_qa_val"]
        elif split == "val":
            names = ["msrvtt_qa_test"]  # ["msrvtt_qa_val"]
        elif split == "test":
            names = ["msrvtt_qa_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self.names = names
        # self.num_frames = 4
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/msrvtt'
        split_files = {
            'train': 'msrvtt_qa_train.jsonl',
            'val': 'msrvtt_qa_test.jsonl',
            'test': 'msrvtt_qa_test.jsonl'
        }
        answer_fp = os.path.join(metadata_dir, 'msrvtt_train_ans2label.json')  # 1500 in total (all classes in train)
        # answer_fp = os.path.join(metadata_dir, 'msrvtt_qa_ans2label.json')  # 4539 in total (all classes in train+val+test)
        answer_clip_id = os.path.join(metadata_dir, 'msrvtt_clip_id.json')
        with open(answer_fp, 'r') as JSON:
            self.ans_lab_dict = json.load(JSON)
        with open(answer_clip_id, 'r') as JSON:
            self.ans_clip_id = json.load(JSON)
        for name in self.names:
            split = name.split('_')[-1]
            target_split_fp = split_files[split]
            # path_or_buf=os.path.join(metadata_dir, target_split_fp)
            metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
            if self.metadata is None:
                self.metadata = metadata
            else:
                self.metadata.update(metadata)
        print("total {} samples for {}".format(len(self.metadata), self.names))
        # data1.update(data2)

    def get_text(self, sample):
        text = sample['question']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return (text, encoding)

    def get_answer_label(self, sample):
        text = sample['answer']
        ans_total_len = len(self.ans_lab_dict) + 1  # one additional class
        try:
            ans_label = self.ans_lab_dict[text]  #
        except KeyError:
            ans_label = -100  # ignore classes
            # ans_label = 1500 # other classes
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores
        # return text, ans_label_vector, scores

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        video_tensor = self.get_video(sample)
        text = self.get_text(sample)
        # index, question_index = self.index_mapper[index]
        qid = index
        if self.split != "test":
            answers, labels, scores = self.get_answer_label(sample)
        else:
            answers = list()
            labels = list()
            scores = list()
            answers, labels, scores = self.get_answer_label(sample)

        return {
            "video": video_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
            "ans_clip_id": self.ans_clip_id,
        }

    def __len__(self):
        return len(self.metadata)