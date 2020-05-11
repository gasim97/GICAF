from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(image, adversarial_image, data_range=255):
    return peak_signal_noise_ratio(image, adversarial_image, data_range=data_range)

def ssim(image, adversarial_image, multichannel=True):
    return structural_similarity(image, adversarial_image, multichannel=multichannel)

def visq(image, adversarial_image):
    pass
