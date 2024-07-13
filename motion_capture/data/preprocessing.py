import torchvision.transforms as image_transforms

# also look at albumenations https://github.com/albumentations-team/albumentations

ImageAugmentations = {
    "BASIC": image_transforms.RandomChoice([
            image_transforms.AugMix(),
            
            image_transforms.RandomChoice([
                image_transforms.ColorJitter(brightness = 0.2),
                image_transforms.ColorJitter(contrast = 0.2),
                image_transforms.ColorJitter(saturation = 0.2),
                image_transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
                image_transforms.ColorJitter(brightness = 0.2, saturation = 0.2),
                image_transforms.ColorJitter(saturation = 0.2, contrast = 0.2),
                image_transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
            ]),
            image_transforms.RandomChoice([
                image_transforms.GaussianBlur(11),
                image_transforms.GaussianBlur(101),
            ]),
            
        ]),
    "AllThePapers": image_transforms.RandomChoice([
            image_transforms.AugMix(),
            image_transforms.AutoAugment(),
            image_transforms.RandAugment(),
        ])
}

