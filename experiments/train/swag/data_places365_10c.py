"""
    separate data loader for imagenet
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms

from data_folder import ImageFolder

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# def loaders(path, batch_size, num_workers, swag_transform_train, swag_transform_test, shuffle_train=True):
def loaders(path, batch_size, num_workers, shuffle_train=True):
    
    train_dir = os.path.join(path, 'train')
    # validation_dir = os.path.join(path, 'validation')
    validation_dir = os.path.join(path, "val")

    # transformations for pretrained models (https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    # test_set =  torchvision.datasets.ImageFolder(validation_dir, transform=transform_test)

    # class_name=['candy_store','classroom','coffee_shop','computer_room','conference_center', 'conference_room', 'lecture_room', 'office', 'supermarket', 'toyshop']
    #[80,92,99,100,101,102,210,244,321,335]

    # class_name = ['classroom','conference_room','office']

    class_name = ['bar','banquet_hall','beer_hall','cafeteria','coffee_shop', 'dining_hall', 'food_court', 'fastfood_restaurant','restaurant_patio','sushi_bar']
            
    # class_name=classes_list
    class_to_idx=dict(zip(class_name,range(len(class_name))))

    print('class_to_idx :',class_to_idx)

    # train_set = torchvision.datasets.ImageFolder(train_dir, transform=swag_transform_train)
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    train_set.class_to_idx=class_to_idx
    # test_set =  torchvision.datasets.ImageFolder(validation_dir, transform=swag_transform_test)
    test_set =  torchvision.datasets.ImageFolder(validation_dir, transform=transform_test)
    test_set.class_to_idx=class_to_idx

    num_classes = len(class_name)

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
        test_set.imgs, # ((img_fn, label),...) for val images imagenames
    )

