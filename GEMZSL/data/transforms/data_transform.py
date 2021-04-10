from torchvision import transforms

def data_transform(name, size=224):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []

    if 'resize_random_crop' in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
        ])
    elif 'resize_center_crop' in name:
        transform.extend(
            transforms.Resize(size),
            transforms.CenterCrop(size),
        )
    elif 'resize_only' in name:
        transform.extend([
            transforms.Resize((size, size)),
        ])
    elif 'resize' in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
        ])
    else:
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])

    if 'colorjitter' in name:
        transform.extend(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2)
        )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform