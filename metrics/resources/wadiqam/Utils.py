"""
PyTorch 1.1 implementation of the following paper:
Bosse S, Maniry D, Müller K R, et al. Deep neural networks for no-reference and full-reference image quality assessment.
IEEE Transactions on Image Processing, 2018, 27(1): 206-219.
    Initial Implementation by Dingquan Li
    Email: dingquanli@pku.edu.cn
    Date: 09/09/2019
------
Sourced and adapted from: https://github.com/lidq92/WaDIQaM
Obtained: 29/05/2020
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
from os.path import dirname

def get_FRnet(weights_path=dirname(__file__) + '/checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRnet(weighted_average=True).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def RandomCropPatches(image, reference=None, patch_size=32, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    im = Image.fromarray(image, mode='RGB')
    try:
      ref = Image.fromarray(reference, mode='RGB')
    except:
      ref = None
    w, h = im.size

    patches = ()
    ref_patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w-patch_size+1)
        h1 = np.random.randint(low=0, high=h-patch_size+1)
        patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
        patches = patches + (patch,)
        if ref is not None:
            ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


def NonOverlappingCropPatches(image, reference=None, patch_size=32):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    im = Image.fromarray(image, mode='RGB')
    try:
      ref = Image.fromarray(reference, mode='RGB')
    except:
      ref = None
    w, h = im.size

    patches = ()
    ref_patches = ()
    stride = patch_size
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
            if ref is not None:
                ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))
                ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)

class FRnet(nn.Module):
    """
    (Wa)DIQaM-FR Model
    """
    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
        super(FRnet, self).__init__()
        self.conv1  = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2  = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4  = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5  = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1_q  = nn.Linear(512*3, 512)
        self.fc2_q  = nn.Linear(512, 1)
        self.fc1_w  = nn.Linear(512*3, 512)
        self.fc2_w  = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)

        h = h.view(-1, 512)

        return h

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):

            h = self.extract_features(x[i])
            h_ref = self.extract_features(x_ref[i])
            h = torch.cat((h - h_ref, h, h_ref), 1)

            h_ = h  # save intermediate features

            h = F.relu(self.fc1_q(h_))
            h = self.dropout(h)
            h = self.fc2_q(h)

            if self.weighted_average:
                w = F.relu(self.fc1_w(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 0.000001 # small constant
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i*n_patches:(i+1)*n_patches] = h

        return q
