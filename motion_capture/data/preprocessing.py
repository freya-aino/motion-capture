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


MainRandomImageTransmuter = image_transforms.RandomChoice([
    # image_transforms.AutoAugment(),

    image_transforms.ColorJitter(brightness = 1),
    image_transforms.ColorJitter(contrast = 1),
    image_transforms.ColorJitter(saturation = 1),
    image_transforms.ColorJitter(brightness = 1, contrast = 1),
    image_transforms.ColorJitter(brightness = 1, saturation = 1),
    image_transforms.ColorJitter(saturation = 1, contrast = 1),
    image_transforms.ColorJitter(brightness = 1, contrast = 1, saturation = 1),

    image_transforms.GaussianBlur(11),
    image_transforms.GaussianBlur(101),

    image_transforms.RandomErasing(scale = (0.1, 0.1)),
    image_transforms.RandomErasing(scale = (0.2, 0.2)),
    image_transforms.RandomErasing(scale = (0.3, 0.3)),
    image_transforms.RandomErasing(scale = (0.4, 0.4)),

    image_transforms.RandomErasing(scale = (0.1, 0.1), value = "random"),
    image_transforms.RandomErasing(scale = (0.2, 0.2), value = "random"),
    image_transforms.RandomErasing(scale = (0.3, 0.3), value = "random"),
    image_transforms.RandomErasing(scale = (0.4, 0.4), value = "random"),

    image_transforms.RandomInvert(p = 1.),
    ])
