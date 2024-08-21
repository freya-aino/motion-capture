from torchvision.transforms import v2

# also look at albumenations https://github.com/albumentations-team/albumentations


ImageAugmentations = {
    "INPLACE": v2.RandomChoice([
        v2.ColorJitter(
            brightness = 0.2, 
            contrast = 0.2, 
            saturation = 0.2,
            hue = 0.2
        ),
        v2.GaussianBlur(
            kernel_size=3,
            sigma = (0.1, 5.0)
        ),
        v2.ElasticTransform(
            alpha = 50,
            sigma = 5
        ),
        v2.RandomPosterize(
            bits = 2,
            p = 1
        ),
        v2.RandomAdjustSharpness(
            sharpness_factor = 2,
            p = 1
        ),
        v2.RandomAutocontrast(
            p = 1
        ),
        v2.Identity(),
        v2.Identity(),
        v2.Identity(),
        v2.Identity(),
    ]),
    "NONE": v2.Identity()
}

