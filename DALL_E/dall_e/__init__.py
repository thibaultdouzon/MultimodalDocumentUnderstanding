from genericpath import isfile
import io, requests
import torch
import torch.nn as nn
import os

from dall_e.encoder import Encoder
from dall_e.decoder import Decoder
from dall_e.utils   import map_pixels, unmap_pixels

def load_model(path: str, device: torch.device = None) -> nn.Module:
    if os.path.isfile(f"models/{os.path.basename(path)}"):
        with open(f"models/{os.path.basename(path)}", "rb") as f:
            model = torch.load(f, map_location=device)
            return model
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            model = torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            model = torch.load(f, map_location=device)
    if isinstance(model, Decoder):
        _ = model.blocks.group_1.upsample
        model.blocks.group_1.upsample = nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)
        _ = model.blocks.group_2.upsample
        model.blocks.group_2.upsample = nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)
        _ = model.blocks.group_3.upsample
        model.blocks.group_3.upsample = nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)
    with open(f"models/{os.path.basename(path)}", "wb") as f:
        torch.save(model, f) 
    return model