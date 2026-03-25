from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
# from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn
import os

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_

import clip

class CrossAttentionLayer(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        
        self.linear_q = nn.Linear(input_size, input_size)
        self.linear_k = nn.Linear(input_size, input_size)
        self.linear_v = nn.Linear(input_size, input_size)
        
        self.dot_product_attention = nn.MultiheadAttention(input_size, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(input_size)
    
    def forward(self, x1, x2):
        q = self.linear_q(x1)
        k = self.linear_k(x2)
        v = self.linear_v(x2)
        
        out, _ = self.dot_product_attention(q, k, v)
        
        out = self.layer_norm(out + x1)
        
        return out


class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True
                 ):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            deit_source = config.get(
                'deit_pretrained',
                "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            )
            if isinstance(deit_source, str) and os.path.isfile(deit_source):
                checkpoint = torch.load(deit_source, map_location="cpu")
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url=deit_source,
                    map_location="cpu",
                    check_hash=True,
                )
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       

        # creat itm head
        text_width = 512
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # creat multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        self.token_mlp = self.build_mlp(input_dim=text_width//2, output_dim=47)

        self.clip_model = clip.load("ViT-B/32")
        self.cross_attention_model = CrossAttentionLayer(input_size=512)

        # self.clip_dim = 32
        self.query_project = nn.Linear(512,512)
        self.value_project = nn.Linear(512,512)
        self.key_project = nn.Linear(512,512)

        self.num_layers = 3

        self.half_mlp = nn.Linear(1024,512)

        self.bbox_improve = nn.Linear(512,512)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        # target_bbox = target_bbox.to('cuda:0')
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text1,text2, fake_image_box, fake_text_pos, alpha=0, is_train=True):
        if is_train:
            # with torch.no_grad():
            #     self.temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            multicls_label, real_label_pos = get_multi_label(label, image) # [B,4]    [3, 5, 6, 8, 9, 11, 16, 20, 21, 23, 24, 25, 26, 30]  
            model = self.clip_model[0]
            preprocess =self.clip_model[1]

            image_embeds = image
            text_embeds1 = clip.tokenize(text1).to(image.device)
            # text_embeds2 = model.text_encoder(input_ids=text2['input_ids'], attention_mask=text2['attention_mask'])
            image_embeds = model.encode_image(image_embeds).float()
            text_embeds = model.encode_text(text_embeds1).float()
            # logits_per_image, logits_per_text = model(image, text_embeds1)

            image_embeds = image_embeds.unsqueeze(0)
            text_embeds = text_embeds.unsqueeze(0)

            image_embeds_ = image_embeds
            text_embeds_ = text_embeds

            for _ in range(self.num_layers):
                image_embeds = self.cross_attention_model(image_embeds,text_embeds_)
                text_embeds = self.cross_attention_model(text_embeds,image_embeds_)
                image_embeds_ = image_embeds
                text_embeds_ = text_embeds


            outputs = torch.cat((image_embeds.squeeze(0), text_embeds.squeeze(0)), dim=1)
            outputs = self.half_mlp(outputs)
            ##================= BIC ========================## 
            # forward the positve image-text pair
            with torch.no_grad():
                bs = image.size(0)          

            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
            itm_labels[real_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
            vl_output = self.itm_head(outputs)   
            loss_BIC = F.cross_entropy(vl_output, itm_labels) 

            ##================= MLC ========================## 
            output_cls = self.cls_head(outputs)
            loss_MLC = F.binary_cross_entropy_with_logits(output_cls, multicls_label.type(torch.float))

            output_coord = self.bbox_head(outputs).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            
            ##================= TMG ========================##    
            token_label = text2.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            token_label[token_label==0] = -100 # -100 index = padding token
            token_label[token_label==1] = 0

            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1

            # tmg_outputs = self.token_mlp(outputs)  # 32,512--->32 32
            B = outputs.shape[0]
            T = token_label.shape[1]
            tmg_outputs = outputs.view(B,-1,2)
            tmg_outputs = self.token_mlp(tmg_outputs.permute(0,2,1)).permute(0,2,1)  # 32 47 2
            loss_TMG = None
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.0)  # -100 index = padding token
            # Only keep active parts of the loss
            attention_mask_rm_CLS = text2.attention_mask[:,1:] # [:,1:] for ingoring class token  32,36
            active_loss = attention_mask_rm_CLS.reshape(-1) == 1  # 32*36
            active_logits = tmg_outputs.reshape(-1, 2)  # 512 2
            active_labels = torch.where(
                active_loss, token_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(token_label)
            )
            loss_TMG = loss_fct(active_logits, active_labels)

            return loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC

        else:
            model = self.clip_model[0]

            image_embeds = image
            text_embeds1 = clip.tokenize(text1).to(image.device)
            # text_embeds2 = model.text_encoder(input_ids=text2['input_ids'], attention_mask=text2['attention_mask'])

            image_embeds = model.encode_image(image_embeds).float()
            text_embeds = model.encode_text(text_embeds1).float()
                        
            image_embeds = image_embeds.unsqueeze(0)
            text_embeds = text_embeds.unsqueeze(0)

            image_embeds_ = image_embeds
            text_embeds_ = text_embeds

            for _ in range(self.num_layers):
                image_embeds = self.cross_attention_model(image_embeds,text_embeds_)
                text_embeds = self.cross_attention_model(text_embeds,image_embeds_)
                image_embeds_ = image_embeds
                text_embeds_ = text_embeds

            outputs = torch.cat((image_embeds.squeeze(0), text_embeds.squeeze(0)), dim=1)
            outputs = self.half_mlp(outputs)

            ##================= BIC ========================## 
            # forward the positve image-text pair
            with torch.no_grad():
                bs = image.size(0)          

            vl_output = self.itm_head(outputs)   

            ##================= MLC ========================## 
            output_cls = self.cls_head(outputs)

            output_coord = self.bbox_head(outputs).sigmoid()
            
            ##================= TMG ========================##
            B = outputs.shape[0]
            T = text2.input_ids.shape[1]-1
            tmg_outputs = outputs.view(B,-1,2)
            tmg_outputs = self.token_mlp(tmg_outputs.permute(0,2,1)).permute(0,2,1)  # 32 47 2
            
            return vl_output, output_cls, output_coord, tmg_outputs   


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

