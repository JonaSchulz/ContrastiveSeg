import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os

from lib.vis.palette import get_cityscapes_colors


class LabelmapVisualizer:
    def __init__(self, configer):
        self.configer = configer
        self.colors = get_cityscapes_colors()
        self.image_names = self.configer.get('labelmap_visualizer', 'image_names')
        self.image_dir = self.configer.get('labelmap_visualizer', 'image_dir')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def save_label(self, data_dict, output):
        image_batch = data_dict['img']
        names_batch = data_dict['name']
        h, w = image_batch.shape[2], image_batch.shape[3]
        output = F.interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)
        logits = []
        for i, name in enumerate(self.image_names):
            if name in names_batch:
                logits.append((output[i], name))

        for logit, name in logits:
            logit = logit.permute(1, 2, 0)
            with torch.no_grad():
                logit = logit.cpu().numpy()
                labelmap = np.asarray(np.argmax(logit, axis=-1), dtype=np.uint8)
            color_image = Image.fromarray(labelmap)
            color_image.putpalette(self.colors)
            iters = self.configer.get('iters')
            color_image.save(self.image_dir + f'/{name}_iter{iters}.png')

