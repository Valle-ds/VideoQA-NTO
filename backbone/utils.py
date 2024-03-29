import warnings
warnings.filterwarnings("ignore")

import os
import urllib
from tqdm import tqdm

import torch
from transformers import BertTokenizer
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from torch.hub import download_url_to_file
from .uniformer2 import uniformerv2_b16, uniformerv2_l14

_MODELS = {
    "base_8": 'https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_b16_8x224.pyth',
    "large_8": 'https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_8x224.pyth',
    "large_16": 'https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_16x224.pyth',
}

def _download(url: str, root: str):
    '''
    Args:
        url (str): download url
        root (str): download root
    '''
    # make directory
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    # check if the file is already downloaded
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # check if files exists
    if os.path.isfile(download_target):
        return download_target
        
    # download the file
    print(f"Downloading {url} to {root}")
    download_url_to_file(url, download_target)

    return download_target


def create_vit(name, pretrained: bool = True, download_root: str = None):
    '''
    Args:
        name (str): name of the model
        pretrained (bool): load pretrained weights
        download_root (str): root directory to download the model
    '''
    if pretrained:
        if name in _MODELS:
            model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/uniformerv2"))
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")

        if 'base' in name:
            model = uniformerv2_b16(model_path = model_path, pretrained=True)
            vision_width = 768
        elif 'large' in name:
            model = uniformerv2_l14(model_path = model_path,pretrained=True)
            vision_width = 1024
    else:
        if 'base' in name:
            model = uniformerv2_b16(pretrained=False)
            vision_width = 768
        elif 'large' in name:
            model = uniformerv2_l14(pretrained=False)
            vision_width = 1024

    return model, vision_width
    

def init_tokenizer(med_config):
    tokenizer = BertTokenizer.from_pretrained(med_config)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,filenames=['/kaggle/working/models/k600_k710_uniformerv2_b16_8x224.pyth', '/kaggle/working/models/pytorch_model.bin']):
    checkpoint = torch.load(filenames[0], map_location='cpu') 

    ww = {}
    for key in checkpoint.keys():
        ww[key.replace('backbone', 'visual_encoder')] = checkpoint[key]

    model.load_state_dict(ww,strict=False)


    checkpoint = torch.load(filenames[1], map_location='cpu') 

    ww_encoder = {}
    for key in model.state_dict().keys():
        key = key.replace('text_encoder', 'bert')
        if key in checkpoint.keys():
            if 'word_embeddings' in key:
                del checkpoint[key]
            else:
                ww_encoder[key.replace('bert', 'text_encoder')] = checkpoint[key]

    model.load_state_dict(ww_encoder,strict=False)

    ww_decoder = {}
    for key in checkpoint.keys():
        ww_decoder[key.replace('bert', 'text_decoder.bert')] = checkpoint[key]

    model.load_state_dict(ww_decoder,strict=False)
    return model
