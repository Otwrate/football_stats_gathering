from  PIL import Image, ImageDraw
import cv2
import torchvision.datasets as Dataset

from extract_data import DatasetBuilder
import numpy as np

def show_bbox_on_image(img, bboxes, shape=None, show=True):
    img1 = ImageDraw.Draw(img)
    bboxes = np.array(bboxes)
    for bbox in np.reshape(bboxes, (-1, 4)):
        img1.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='blue')
    img.show()





if __name__ =='__main__':
    train_path = r'datatsets/pedestrians/Train/Train/'
    train_dataset = Dataset.ImageFolder(train_path, )
    train = DatasetBuilder(train_path)
    I, labels = train[100]
    bboxes = labels['boxes']
    show_bbox_on_image(I, bboxes)
    pass