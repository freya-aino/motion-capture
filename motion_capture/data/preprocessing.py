import torchvision.transforms as image_transforms

# -------------- TRANSMUTE DATA --------------

# flip image
# flip / rotate image

# remove croped rectangle
# remove croped borders
# swap parts of the image
# collect RL backgrounds to insert into test images (my room)

# swap croped rectangle
# add black border extending the image
# add black border croping the image
# collect RL backgrounds to insert into test images (my room)


# also look at albumenations https://github.com/albumentations-team/albumentations

ImagePertubators = {
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

