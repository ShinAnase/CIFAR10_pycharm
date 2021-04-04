import albumentations as albu

def get_transform():
    train_transform = [
        #albu.HorizontalFlip(p=1),
        #albu.VerticalFlip(p=1),
        #albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
        #albu.CenterCrop(height=24, width=24, p=1.0),
        #albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        #albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    ]
    return albu.Compose(train_transform)


def get_augmentation_Train():
    transform1 = []
    transform2 = [
        albu.HorizontalFlip(p=1),
        #albu.VerticalFlip(p=1),
        # albu.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=1),
        # albu.CenterCrop(height=24, width=24, p=1.0),
        # albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        # albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    ]
    # transform3 = [
    #    #albu.HorizontalFlip(p=1),
    #    #albu.VerticalFlip(p=1),
    #    albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
    #    #albu.CenterCrop(height=24, width=24, p=1.0),
    #    #albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
    #    #albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    # ]

    transformsCompo = [
        albu.Compose(transform1),
        #albu.Compose(transform2),
        # albu.Compose(transform3),
    ]

    return transformsCompo


def get_augmentation_TTA():
    transform1 = [
        # albu.HorizontalFlip(p=1),
        # albu.VerticalFlip(p=1),
        # albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
        # albu.CenterCrop(height=24, width=24, p=1.0),
        # albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        # albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    ]
    transform2 = [
        #albu.HorizontalFlip(p=1),
        albu.VerticalFlip(p=1),
        #albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
        # albu.CenterCrop(height=24, width=24, p=1.0),
        # albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        # albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    ]
    # transform3 = [
    #    #albu.HorizontalFlip(p=1),
    #    #albu.VerticalFlip(p=1),
    #    albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
    #    #albu.CenterCrop(height=24, width=24, p=1.0),
    #    #albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
    #    #albu.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value = 255.0, p=1.0),
    # ]

    transformsCompo = [
        albu.Compose(transform1),
        #albu.Compose(transform2),
        # albu.Compose(transform3),
    ]

    return transformsCompo