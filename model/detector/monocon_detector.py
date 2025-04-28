import os
import sys
import torch
import torch.nn as nn

from typing import Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from model import DLA, DLAUp, MonoConDenseHeads


default_head_config = {
    'num_classes': 3,
    'num_kpts': 9,
    'num_alpha_bins': 12,
    'max_objs': 30,
}


default_test_config = {
    'topk': 30,
    'local_maximum_kernel': 3,
    'max_per_img': 30,
    'test_thres': 0.4,
}



from transformers import CLIPVisionModel, CLIPImageProcessor

 

class ClipVisionBackbone(nn.Module):

    def __init__(self, model_name="openai/clip-vit-base-patch32"):

        super().__init__()

        self.clip_model = CLIPVisionModel.from_pretrained(model_name)

        self.clip_processor = CLIPImageProcessor.from_pretrained(model_name)

       

        self.output_dim = self.clip_model.config.hidden_size  # 768 for ViT-B/32

       

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        """

        img: Tensor of shape (B, 3, H, W), pixel values in [0,1] range.

        """

        # Preprocess: resize, normalize (inside processor)

        processed = self.clip_processor(images=img, return_tensors="pt")

        input_img = processed['pixel_values'].to(img.device)

       

        # Forward through CLIP vision encoder

        outputs = self.clip_model(pixel_values=input_img)

       

        # Remove CLS token and reshape

        features = outputs.last_hidden_state[:, 1:, :]  # (B, N, C)

       

        B, N, C = features.shape

        feat_size = int(N ** 0.5)

        features = features.permute(0, 2, 1).view(B, C, feat_size, feat_size)  # (B, C, H', W')

       

        return features


class MonoConDetector(nn.Module):
    def __init__(self,
                 num_dla_layers: int = 34,
                 pretrained_backbone: bool = True,
                 head_config: Dict[str, Any] = None,
                 test_config: Dict[str, Any] = None):
        
        super().__init__()
        
        self.backbone = ClipVisionBackbone(model_name="openai/clip-vit-base-patch32") #self.neck = None # No neck needed head_in_ch = self.backbone.output_dim # 768


        # self.backbone = DLA(num_dla_layers, pretrained=pretrained_backbone)
        # self.neck = DLAUp(self.backbone.get_out_channels(start_level=2), start_level=2)
        self.neck = None  # No neck needed

        if head_config is None:
            head_config = default_head_config
        if test_config is None:
            test_config = default_test_config
            
        if num_dla_layers in [34, 46]:
            head_in_ch = 64
        else:
            head_in_ch = 128
            
        self.head = MonoConDenseHeads(in_ch=self.backbone.output_dim, test_config=test_config, **head_config)
        
        
    def forward(self, data_dict: Dict[str, Any], return_loss: bool = True) -> Tuple[Dict[str, torch.Tensor]]:
        
        feat = self._extract_feat_from_data_dict(data_dict)
        
        if self.training:
            pred_dict, loss_dict = self.head.forward_train(feat, data_dict)
            if return_loss:
                return pred_dict, loss_dict
            return pred_dict
        
        else:
            pred_dict = self.head.forward_test(feat)
            return pred_dict
        
    
    def batch_eval(self, 
                   data_dict: Dict[str, Any], 
                   get_vis_format: bool = False) -> Dict[str, Any]:
        
        if self.training:
            raise Exception(f"Model is in training mode. Please use '.eval()' first.")
        
        pred_dict = self.forward(data_dict, return_loss=False)
        eval_format = self.head._get_eval_formats(data_dict, pred_dict, get_vis_format=get_vis_format)
        return eval_format
    
    
    def load_checkpoint(self, ckpt_file: str):
        model_dict = torch.load(ckpt_file)['state_dict']['model']
        self.load_state_dict(model_dict)


    def _extract_feat_from_data_dict(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        img = data_dict['img']
        return self.neck(self.backbone(img))[0]
