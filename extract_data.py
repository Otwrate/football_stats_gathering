import glob
import os
import numpy as np
from logging import warn, warning
from pathlib import PurePath, Path

import torch
import torchvision
import torchvision.datasets as Dataset
import xml.etree.ElementTree as ET
import pandas as pd


class DatasetBuilder(torch.utils.data.Dataset):
    def __init__(self, path=None, annotation_path=None, dataset=None, mapper={'person': 0,
                                                                              'hand': 1,
                                                                              'foot': 2,
                                                                              'person': 3,
                                                                              'head': 4,
                                                                              'person-like': 5
                                                                              }, transforms=None):

        self.transforms = transforms
        self.annotation_path = os.path.join(path, 'Annotations')
        self.data_path = path
        self.labels = pd.DataFrame(columns=['filename',
                                            'width',
                                            'height',
                                            'class_id',
                                            'name',
                                            'pose',
                                            'truncated',
                                            'occluded',
                                            'bndbox_xmin',
                                            'bndbox_ymin',
                                            'bndbox_xmax',
                                            'bndbox_ymax',
                                            ])
        self.dataset = dataset
        self.mapper = mapper
        self.pedestrians = None
        self.__get_dataset()

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        img_idx = self.dataset.samples[idx][0].split('\\')[-1].strip('.jpg')
        pedestrians = self.pedestrians.loc[img_idx]
        targets = []
        items = pedestrians.reset_index().T.to_dict()
        for k, v in items.items():
            targets.append({
                # 'boxes': {'bndbox_xmin': torch.as_tensor(np.float32(v['bndbox_xmin']), dtype=torch.float32),
                #           'bndbox_ymin': torch.as_tensor(np.float32(v['bndbox_ymin']), dtype=torch.float32),
                #           'bndbox_xmax': torch.as_tensor(np.float32(v['bndbox_xmax']), dtype=torch.float32),
                #           'bndbox_ymax': torch.as_tensor(np.float32(v['bndbox_ymax']), dtype=torch.float32)},
                'boxes': torch.as_tensor(np.array([(np.float32(v['bndbox_xmin'])),
                                                    (np.float32(v['bndbox_ymin'])),
                                                   (np.float32(v['bndbox_xmax'])),
                                                    (np.float32(v['bndbox_ymax']))])),
                'lables': 1})
        bboxes = pedestrians[['bndbox_xmin', 'bndbox_ymin', 'bndbox_xmax', 'bndbox_ymax']].astype(np.float32).values
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        occlusion = pedestrians['occluded']  # .astype(np.float32)
        img = np.array(img)
        try:
            occlusion = torch.as_tensor(occlusion, dtype=torch.float32)
        except TypeError:
            occlusion = torch.zeros(bboxes.shape[0])
        # target['boxes'] = bboxes
        # target['occlusion'] = occlusion
        if self.transforms is not None:
            img, target = self.transforms(img, targets)
        return torch.from_numpy(np.array(img)).reshape([img.shape[2], img.shape[1], img.shape[0]]), targets

    def __len__(self):
        return len(self.dataset)

    def __get_data(self):
        self.dataset = torchvision.datasets.ImageFolder(self.data_path)

    def __get_annotations(self):
        for annotation in os.listdir(self.annotation_path):
            tree = ET.parse(os.path.join(self.annotation_path, annotation))
            self._get_labels(tree)
        self.labels = self.labels.set_index('filename')
        self.pedestrians = self.labels[(self.labels.name == 'person') | (self.labels.name == 'person-like')]
        return self.labels

    def _get_labels(self, annotation):
        def extract_bb(object):
            bndbox_xmin = object.find('bndbox').find('xmin').text
            bndbox_ymin = object.find('bndbox').find('ymin').text
            bndbox_xmax = object.find('bndbox').find('xmax').text
            bndbox_ymax = object.find('bndbox').find('ymax').text
            return bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax

        file_name = annotation.find('filename').text.strip('.xml')
        width = annotation.find('size').find('width').text
        height = annotation.find('size').find('height').text
        # try:
        for object in annotation.getroot().iter('object'):
            name = object.find('name').text
            pose = object.find('pose').text
            truncated = object.find('truncated').text
            try:
                occluded = np.float32(object.find('occluded').text)
            except AttributeError:
                occluded = None
            bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax = extract_bb(object)
            self.labels = self.labels.append({'filename': file_name,
                                              'name': name,
                                              'class_id': self.mapper[name],
                                              'pose': pose,
                                              'width': width,
                                              'height': height,
                                              'truncated': truncated,
                                              'occluded': occluded,
                                              'bndbox_xmin': bndbox_xmin,
                                              'bndbox_ymin': bndbox_ymin,
                                              'bndbox_ymax': bndbox_ymax,
                                              'bndbox_xmax': bndbox_xmax,
                                              }, ignore_index=True)
        for part in object.iter('part'):
            name = part.find('name').text
            bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax = extract_bb(part)
            self.labels = self.labels.append({'filename': file_name,
                                              'name': name,
                                              'class_id': self.mapper[name],
                                              'bndbox_xmin': bndbox_xmin,
                                              'bndbox_ymin': bndbox_ymin,
                                              'bndbox_xmax': bndbox_xmax,
                                              'bndbox_ymax': bndbox_ymax,
                                              }, ignore_index=True)

        # except AttributeError as E:
        #     print(E)

    def __get_dataset(self):
        _ = self.__get_annotations()
        self.__get_data()
        pass


if __name__ == '__main__':
    train_path = r'datatsets/pedestrians/Train/Train/'
    train_dataset = Dataset.ImageFolder(train_path, )
    train = DatasetBuilder(train_path)
    train.__getitem__(3)
    pass
