from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed

import os
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_

from transformers import ViltProcessor, ViltModel, ViltForTokenClassification

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
            deit_ckpt = os.environ.get(
                "DEIT_CKPT",
                "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            )
            if deit_ckpt.startswith("http://") or deit_ckpt.startswith("https://"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    url=deit_ckpt, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(deit_ckpt, map_location="cpu")
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        
        # text_width = self.text_encoder.config.hidden_size
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        text_width = 768
        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # creat multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        self.token_mlp = self.build_mlp(input_dim=text_width//2, output_dim=39)

        self.processor = ViltProcessor.from_pretrained("./DGM4/vilt_b32_mlm")

        self.token_model = ViltForTokenClassification.from_pretrained("./DGM4/vilt_b32_mlm")

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
        target_bbox = target_bbox.to('cuda:0')
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

    def forward(self, image, inputs, label, fake_image_box, fake_text_pos, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            multicls_label, real_label_pos = get_multi_label(label,image.pixel_values) # [B,4]    [3, 5, 6, 8, 9, 11, 16, 20, 21, 23, 24, 25, 26, 30]  
            
            outputs = self.token_model.vilt(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,pixel_values=image.pixel_values)

            # ViLT
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            outputs = last_hidden_state[:,0,:] # 32,768

            ##================= BIC ========================## 
            # forward the positve image-text pair
            with torch.no_grad():
                bs = image.pixel_values.size(0)          

            itm_labels = torch.ones(bs, dtype=torch.long).to("cuda:0")
            itm_labels[real_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
            vl_output = self.itm_head(outputs)   
            loss_BIC = F.cross_entropy(vl_output, itm_labels) 

            ##================= MLC ========================## 
            output_cls = self.cls_head(outputs)
            loss_MLC = F.binary_cross_entropy_with_logits(output_cls, multicls_label.type(torch.float))

            output_coord = self.bbox_head(outputs).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            
            ##================= TMG ========================##    
            token_label = inputs.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            token_label[token_label==0] = -100 # -100 index = padding token
            token_label[token_label==1] = 0
            

            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1
            

            token_cls_output = self.token_model(
                    input_ids=inputs.input_ids[:,1:], 
                    attention_mask=inputs.attention_mask[:,1:], 
                    pixel_values=image.pixel_values,  # Provide the image embeddings as pixel_values
                    pixel_mask=image.pixel_mask,
                    return_dict=True,  # For classification loss if needed
                    labels = token_label, 
                )
            loss_TMG = token_cls_output.loss

            return loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC

        else:

            outputs = self.token_model.vilt(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,pixel_values=image.pixel_values)

            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            outputs = last_hidden_state[:,0,:] # 32,768

            ##================= BIC ========================## 
            # forward the positve image-text pair
            with torch.no_grad():
                bs = image.pixel_values.size(0)             

            vl_output = self.itm_head(outputs)   

            ##================= MLC ========================## 
            output_cls = self.cls_head(outputs)

            output_coord = self.bbox_head(outputs).sigmoid()
            
            ##================= TMG ========================##    
            
            token_cls_output = self.token_model(
                    input_ids=inputs.input_ids[:,1:], 
                    attention_mask=inputs.attention_mask[:,1:], 
                    pixel_values=image.pixel_values,  # Provide the image embeddings as pixel_values
                    pixel_mask=image.pixel_mask,
                    return_dict=True,  # For classification loss if needed
                )
            token_cls_output = token_cls_output.logits
            return vl_output, output_cls, output_coord, token_cls_output 


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

