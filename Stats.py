from skimage.measure import compare_ssim, compare_psnr

def psnr(image, adversarial_image, data_range=255):
    return compare_psnr(image, adversarial_image, data_range=data_range)

def ssim(image, adversarial_image, multichannel=True):
    return compare_ssim(image, adversarial_image, multichannel=multichannel)

def visq(image, adversarial_image):
    pass
