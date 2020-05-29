from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from gicaf.metrics.WaDIQaM import WaDIQaM
from cv2 import cvtColor, COLOR_BGR2RGB

def psnr(image, adversarial_image, model_metadata):
    return peak_signal_noise_ratio(adversarial_image,
                                    image, 
                                    data_range=(model_metadata['bounds'][1] - model_metadata['bounds'][0]))

def ssim(image, adversarial_image, model_metadata):
    return structural_similarity(adversarial_image, 
                                image, 
                                multichannel=True if model_metadata['channels'] > 1 else False)

def wadiqam(image, adversarial_image, model_metadata):
    data = WaDIQaM.NonOverlappingCropPatches(cvtColor(adversarial_image, COLOR_BGR2RGB) if model_metadata['bgr'] else adversarial_image,
                                            cvtColor(image, COLOR_BGR2RGB) if model_metadata['bgr'] else image)
    model = WaDIQaM.get_FRnet()
    dist_patches = data[0].unsqueeze(0)
    ref_patches = data[1].unsqueeze(0)
    score = model((dist_patches, ref_patches))
    return score.item()
