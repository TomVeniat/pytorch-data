import os

import errno
import torch
import torch.utils.data as data

import numpy as np

from PIL import Image
from tqdm import tqdm


class PartLabels(data.Dataset):
    """Pytorch integration of the Part Labels Database from http://vis-www.cs.umass.edu/lfw/part_labels/"""

    data_urls = [
        'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz',
        'http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz'
    ]

    text_urls = [
        'http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt',
        'http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt',
        'http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt'
    ]

    raw_folder = 'part-labels-raw'
    processed_folder = 'part-labels'

    images_folder = 'lfw_funneled'
    labels_folder = 'parts_lfw_funneled_gt_images'

    train_list_file = 'parts_train.txt'
    val_list_file = 'parts_validation.txt'
    test_list_file = 'parts_test.txt'

    processed_training_file = 'training.pt'
    processed_test_file = 'test.pt'
    processed_val_file = 'val.pt'

    def __init__(self, root, train=False, validation=False, test=False, transform=None,
                 target_transform=None, joint_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.val = validation
        self.test = test
        self.joint_transform = joint_transform

        if download:
            self.prepare()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(os.path.join(root, self.processed_folder, self.processed_training_file))
        elif self.val:
            self.val_data, self.val_labels = torch.load(os.path.join(root, self.processed_folder, self.processed_val_file))
        elif self.test:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.processed_test_file))
        else:
            raise AttributeError('At least one of train, validation or test init flag should be True.')

    def __getitem__(self, index):
        if self.train:
            input, target = self.train_data[index], self.train_labels[index]
        elif self.val:
            input, target = self.val_data[index], self.val_labels[index]
        else:
            input, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        input = Image.fromarray(np.transpose(input.numpy(), (1, 2, 0)))
        target = Image.fromarray(np.transpose(target.numpy(), (1, 2, 0)))

        if self.joint_transform is not None:
            input, target = self.joint_transform(input, target)

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        if self.train:
            return 1500
        elif self.val:
            return 500
        else:
            return 927

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_val_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_test_file))

    def prepare(self):

        if self._check_exists():
            print('Files already downloaded')
            return

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            print(e)
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        self.fetch_data(self.data_urls + self.text_urls)

        import tarfile
        for url in self.data_urls:

            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            print('Extracting {}'.format(filename))
            with tarfile.open(file_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, os.path.join(self.root,self.raw_folder))
            os.unlink(file_path)

        print('Processing')

        self.load_save()

        print('Done!')

    def get_set(self, list_file, set_name):
        infos_location = os.path.join(self.root, self.raw_folder, list_file)

        image_paths = get_filepaths(infos_location)
        images_location = os.path.join(self.root, self.raw_folder, self.images_folder)
        images = get_images(image_paths, prefix=images_location, extension='.jpg', name='{} images'.format(set_name))

        label_paths = get_filepaths(infos_location, prefix_folder=False)
        labels_location = os.path.join(self.root, self.raw_folder, self.labels_folder)
        labels = get_images(label_paths, prefix=labels_location, extension='.ppm', name='{} labels'.format(set_name))

        return torch.stack(images), torch.stack(labels)

    def load_save(self):
        sets = [(self.train_list_file, self.processed_training_file, 'Train'),
                (self.val_list_file, self.processed_val_file, 'Validation'),
                (self.test_list_file, self.processed_test_file, 'Test')]

        for samples_list, dest_file, set_name in sets:
            set_data = self.get_set(samples_list, set_name)
            with open(os.path.join(self.root, self.processed_folder, dest_file), 'wb') as f:
                torch.save(set_data, f)

    def fetch_data(self, url_list):
        import urllib
        for url in url_list:
            print('Downloading ' + url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            urllib.request.urlretrieve(url, file_path)


def get_filepaths(path, prefix_folder=True):
    with open(path, 'rb') as f:
        res = []
        lines = f.readlines()
        for l in lines:
            data = l.decode().split()
            filename = data[0] + '_' + data[1].zfill(4)
            if prefix_folder:
                filename = os.path.join(data[0], filename)
            res.append(filename)
    return res


def get_images(image_names, prefix='', extension='', name='Loading data'):
    images = []
    for img_name in tqdm(image_names, desc=name):
        img_path = os.path.join(prefix, img_name + extension)
        with Image.open(img_path) as pil_img:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_img.tobytes()))
            img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            images.append(img)
    return images
