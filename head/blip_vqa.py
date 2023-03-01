from .med import BertConfig, BertModel, BertLMHeadModel

# from models.blip import create_vit, init_tokenizer, load_checkpoint
from backbone.utils import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np


class BLIP_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'bert-base-multilingual-uncased',  
                 vit = 'base_8',   
                 freeze_vit = False          
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, pretrained=False)
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        self.vision_width = vision_width
        self.tokenizer = init_tokenizer(med_config)  
        
        encoder_config = BertConfig.from_pretrained(med_config)
        encoder_config.encoder_width = vision_width
        encoder_config.vocab_size = encoder_config.vocab_size + 2
        encoder_config.num_attention_heads = 12
        encoder_config.num_hidden_layers = 12
        encoder_config.max_position_embeddings = 512
        encoder_config.add_cross_attention = True
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        decoder_config = BertConfig.from_pretrained(med_config)        
        decoder_config.vocab_size = decoder_config.vocab_size + 2
        decoder_config.encoder_width = vision_width
        decoder_config.num_attention_heads = 12
        decoder_config.num_hidden_layers = 12
        decoder_config.max_position_embeddings = 512
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        self.text_decoder = BertLMHeadModel(config=decoder_config)          


    def forward(self, video, question, answer=None, n=None, weights=None, train=True, inference='generate', k_test=128):
        image_embeds = self.visual_encoder(video)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(video.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(video.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(video.device) 
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n                
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            
            loss = weights * answer_output.loss
            loss = loss.sum()/video.size(0)

            return loss
            

        else: 
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            if inference=='generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((video.size(0),1),fill_value=self.tokenizer.bos_token_id,device=video.device)
                
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers
            
            elif inference=='rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) 
                return max_ids
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids
    
    
def blip_vqa(pretrained=False, filenames=[],**kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
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


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
