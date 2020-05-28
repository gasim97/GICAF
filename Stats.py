from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from gicaf.metrics.WaDIQaM import WaDIQaM
from cv2 import cvtColor, COLOR_BGR2RGB

def psnr(image, adversarial_image, data_range=255):
    return peak_signal_noise_ratio(image, adversarial_image, data_range=data_range)

def ssim(image, adversarial_image, multichannel=True):
    return structural_similarity(image, adversarial_image, multichannel=multichannel)

def wadiqam(image, adversarial_image, bgr=False):
    data = WaDIQaM.NonOverlappingCropPatches(cvtColor(image, COLOR_BGR2RGB) if bgr else image, 
                                            cvtColor(adversarial_image, COLOR_BGR2RGB) if bgr else adversarial_image)
    model = WaDIQaM.get_FRnet()
    dist_patches = data[0].unsqueeze(0)
    ref_patches = data[1].unsqueeze(0)
    score = model((dist_patches, ref_patches))
    return score.item()
