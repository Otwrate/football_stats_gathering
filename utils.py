from  PIL import Image, ImageDraw
import cv2
import torchvision.datasets as Dataset

from extract_data import DatasetBuilder
import numpy as np

def show_bbox_on_image(img, bboxes, shape=None, show=True):
    # for bbox in np.reshape(bboxes, (-1, 4)):
    #     img = cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[0], bbox[1]), (255, 0 , 0))
    # if show:
    #     cv2.imshow('bboxes', img)
    #     cv2.waitKey(10000)

    img1 = ImageDraw.Draw(img)
    bboxes = np.array(bboxes)
    for bbox in np.reshape(bboxes, (-1, 4)):
        img1.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='blue')
    img.show()


# def scale_bb(img, targets):
#     width, height = img.shape[0], img.shape[1]
#     for bbox in targets['boxes']:




if __name__ =='__main__':
    train_path = r'datatsets/pedestrians/Train/Train/'
    train_dataset = Dataset.ImageFolder(train_path, )
    train = DatasetBuilder(train_path)
    I, labels = train[100]
    # path = r'C:\Users\otwra\PycharmProjects\football_stats_gathering\datatsets\pedestrians\Test\Test\JPEGImages\image (2).jpg'
    # I = cv2.imread(path)
    bboxes = labels[['bndbox_xmin', 'bndbox_ymin', 'bndbox_xmax', 'bndbox_ymax']].astype(np.float32).values
    bboxes = labels['boxes']
    # shape = labels[['width', 'height']].astype(np.float32).values[0]
    # cv2.imshow('raw', I)
    show_bbox_on_image(I, bboxes)
    pass