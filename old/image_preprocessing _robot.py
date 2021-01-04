import h5py
import os
from PIL import Image
from torchvision import transforms
import config
import numpy
from tqdm import tqdm

try:
    import cPickle as pickle
except:
    import pickle

transform = transforms.Compose([
    transforms.Scale(config.image_scale),
    transforms.CenterCrop(config.target_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def image_preprocessing(images_path, target_path, image_size, processed_image_path, img2idx_path):
    num_of_pics = len(os.listdir(images_path))

    features_shape = (num_of_pics, 3, image_size, image_size)
    img2idx = {}
    with h5py.File(processed_image_path, 'w', libver='latest') as f:
        images = f.create_dataset('images', shape=features_shape, dtype='float16')

        i = 0
        for image_name in tqdm(os.listdir(images_path)):
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images[i, :, :] = image.numpy().astype('float16')
            img2idx[image_name] = i
            i += 1

    with open(img2idx_path, 'wb') as f:
        pickle.dump(img2idx, f)


if __name__ == '__main__':
    image_preprocessing(config.images_path_train, config.train_target_path,
                        config.target_size,
                        config.preprocessed_image_train, config.img2idx_train)
