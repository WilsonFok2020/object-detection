import torch
import numpy as np
from collections import defaultdict
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

TORCH_DATATYPE = np.float32

class Cast_to_Pytorch(object):
    
    def __init__(self, level_max):
        self.max = level_max
    def __call__(self, x):
        return torch.from_numpy(x.astype(TORCH_DATATYPE) / self.max).permute(2,0,1).contiguous()


def load_classes(csv_reader):
    '''
    a very simple way to read the files
    Pandas may be used to do more complicated operations
    '''
    
    result = {}
    for line, row in enumerate(csv_reader):
        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError('line {}: format should be classname, class id'.format(class_name,
                             class_id))
        try:
            class_id = int(class_id)
        except TypeError as e:
            raise TypeError('{} \n line {} : malformed class id: {}'.format(
                    e, line, class_id))
            
        if class_name in result:
            raise ValueError('class names must be unique. \n line {} : duplicate class name : {}'.format(
                    line, class_name))
            
        result[class_name] = class_id
        
    return result

class CSVDataset(Dataset):
    
    def __init__(self, train_file, class_list, color_list, transform=None):
        self.train_file = train_file
        self.class_list = class_list
        self.color_list = color_list
        self.transform = transform
        
        try:
            with open(self.class_list, 'r', newline='') as file:
                self.classes = load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError('invalue csv class file {} {}'.format(self.class_list, e))
            
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.color_labels = {}
        try:
            with open(self.color_list, 'r', newline='') as file:
                self.colors = load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError('invalid CSV color file: {} {}'.format(self.color_list, e))
            
        for key, value in self.colors.items():
            self.color_labels[value] = key
            
        try:
            with open(self.train_file, 'r', newline='') as file:

                # csv with img_path, x1, y1, x2, y2, class_name
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes, self.colors)
        except ValueError as e:
            raise ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e))
            
        self.image_names = list(self.image_data.keys())

    

   

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            # only apply transformation on the entire images
            img = self.transform(sample['img'])
            sample['img'] = img

        return sample

    def load_image(self, image_index):
        '''
        save all the image data in npy format for easy manipulation
        '''
        filepath = self.image_names[image_index]
        try:
            img = np.load(filepath)
        except FileNotFoundError:
            raise FileNotFoundError('npy filepath in csv may be wrong {}'.format(filepath))
        return img

    def load_annotations(self, image_index):
        L = 6 # x1,y1,x2,y2, shape, color
        
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, L))


        # parse annotations
        for _, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                print ('may have orientation error?')
                continue

            annotation        = np.zeros((1, L))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotation[0, 5]  = self.color_name_to_labels(a['color'])
            annotations       = np.append(annotations, annotation, axis=0)

        return torch.from_numpy(annotations.astype(TORCH_DATATYPE))

    def _read_annotations(self, csv_reader, classes, colors):
        result = defaultdict(list)
        
        for line, row in enumerate(csv_reader):
            

            try:
                img_file, x1, y1, x2, y2, class_name, color_name = row
            except ValueError:
                raise ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name,color_name\' or \'img_file,,,,,\''.format(line))

            
            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name, color_name) == ['']*6:
                raise ValueError('line contains no annotation. line: {}'.format(line))

            x1, y1, x2, y2 = [int(item) for item in (x1, y1, x2, y2)]

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
            if color_name not in colors:
                raise ValueError('line {}: unknown color name: \'{}\' (colors: {})'.format(line, color_name, colors))
                

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'color': color_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]
    def color_name_to_labels(self, name):
        return self.colors[name]
    

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values())+1