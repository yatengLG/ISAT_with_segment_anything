"""Module providing functions interacting with ISAT annotation files"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level
import os  # interact with the operating system
import json  # manipulate json files
import cv2  # OpenCV

image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']  # supported image types


class Anything2ISAT:
    """
    Convert any annotation format to ISAT json format

    The ISAT (Image Segmentation Annotation Tool) format provides a structured approach for representing image annotations
    File Naming: Each image has a corresponding .json file named after the image file (without the image extension)

    ['info']: Contains metadata about the dataset and image
        ['description']: Always 'ISAT'
        ['folder']: The directory where the images are stored
        ['name']: The name of the image file
        ['width'], ['height'], ['depth']: The dimensions of the image; depth is assumed to be 3 for RGB images
        ['note']: An optional field for any additional notes related to the image
   ['objects']: Lists all the annotated objects in the image
        ['category']: The class label of the object. If the category_id from MSCOCO does not have a corresponding entry, 'unknown' is used
        ['group']: An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number. Group numbering starts at 1
        ['segmentation']: A list of [x, y] coordinates forming the polygon around the object
        ['area']: The area covered by the object in pixels
        ['layer']: A float indicating the sequence of the object. It increments within the same group, starting at 1.0
        ['bbox']: The bounding box coordinates in the format [x_min, y_min, x_max, y_max]
        ['iscrowd']: A boolean value indicating if the object is part of a crowd

    Required input:
        image_dir  # a folder that stores all images
        annotations_dir  # a folder that stores the annotations file(s)

    Output:
    ISAT format json files in under images_dir
    """
    def __init__(self,
                 images_dir: str = None,
                 annotations_dir: str = None,
                 output_dir: str = None):
        self.images_dir = images_dir  # images directory
        self.annotations_dir = annotations_dir  # annotations directory
        self.output_dir = output_dir  # ISAT json output directory

        os.makedirs(self.output_dir, exist_ok=True)  # create output directory

    def from_custom(self, objectes_dictionary):
        """
        Convert custom format to ISAT format
        ['info'] is generated from images directory
        ['objects'] requires objectes_dictionary - a dictionary of the objects of all images, e.g. use .get('image0', []) to get the objects of image0
        """
        image_names = [name for name in os.listdir(self.images_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # get all image names
        image_paths = [os.path.join(self.images_dir, name) for name in image_names]  # get all image paths
        for image_path in image_paths:
            image = cv2.imread(image_path)  # load the image in BRG scale
            image_basename = os.path.basename(image_path)  # get the basename
            dataset = {}  # to store dataset information
            dataset['info'] = {}
            dataset['info']['description'] = 'ISAT'  # it must be 'ISAT'
            dataset['info']['folder'] = self.images_dir  # image dir
            dataset['info']['name'] = image_basename  # image name
            dataset['info']['width'] = image.shape[1]  # image width
            dataset['info']['height'] = image.shape[0]  # image height
            dataset['info']['depth'] = image.shape[2]  # image depth
            dataset['info']['note'] = ''
            dataset['objects'] = objectes_dictionary.get(image_basename, [])  # get objects info
            json_path = os.path.splitext(image_path)[0] + '.json'  # output json path
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(dataset, file, indent=4)  # save the ISAT format json file
        return None

    def bbox_within(self, bbox_1, bbox_2):  # 这个函数查看两个物体的边框，如果是包含关系的话分到一个组
        """Check if two objects belong to the same group"""
        return all(bbox_1[idx] >= bbox_2[idx] for idx in [0, 1]) and all(bbox_1[idx] <= bbox_2[idx] for idx in [2, 3])

    def from_yolo_seg(self, class_dictionary=None):
        """
        Get the objects information form YOLO segmentation txt file
            Key differences:
            1. 'segmentation'
                YOLO: [x1,  y1, x2, y2, ..., xn, yn]
                ISAT: [[x1, y1], [x2, y2], ..., [xn, yn]]
            3. layer
            4. group
        """
        def yolo2isat_segmentation(yolo_seg, img_width, img_height):
            """Convert YOLO segmentation format to ISAT segmentation format"""
            return [[round(x * img_width), round(y * img_height)] for x, y in zip(yolo_seg[::2], yolo_seg[1::2])]

        def get_isat_bbox(segmentation):
            """Calculate the bbox from the ISAT segmentation"""
            xs = [point[0] for point in segmentation]  # x-coordinates
            ys = [point[1] for point in segmentation]  # y-coordinates
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        image_names = [name for name in os.listdir(self.images_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # get all image names
        image_paths = [os.path.join(self.images_dir, name) for name in image_names]  # get all image paths
        annotation_names = [name for name in os.listdir(self.annotations_dir) if any(name.lower().endswith(file_type) for file_type in ['txt'])]  # get annotation file names
        annotation_paths = [os.path.join(self.annotations_dir, name) for name in annotation_names]  # get annotation paths
        for idx, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)  # load the image in BRG scale
            image_width, image_height = image.shape[1], image.shape[0]  # get the image dimensions
            isat_objects = []
            groups, layer = [], 1.0  # initialize layer as a floating point number
            with open(annotation_paths[idx], 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.split()  # split each line
                    class_index = int(parts[0])  # get the class index
                    yolo_segmentation = list(map(float, parts[1:]))  # get the yolo_segmentation
                    isat_segmentation = yolo2isat_segmentation(yolo_segmentation, image_width, image_height)  # convert yolo_segmentation to isat_segmentation
                    bbox = get_isat_bbox(isat_segmentation)  # calculate the bbox from segmentation
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # roughly calculate the bbox area as segmentation area, it will be replaced anyway
                    group = next((g for g, g_bbox in enumerate(groups, 1) if self.bbox_within(bbox, g_bbox)), len(groups) + 1)
                    if group == len(groups) + 1:
                        groups.append(bbox)
                    isat_objects.append({
                        'category': class_dictionary.get(class_index, 'unknown'),  # for example {0: 'class 0', 1: 'class 1'}
                        'group': group,  # group increases if the bbox is not within another
                        'segmentation': isat_segmentation,
                        'area': area,
                        'layer': layer,  # Increment layer for each object
                        'bbox': bbox,
                        'iscrowd': False,
                        'note': ''
                    })
                    layer += 1.0  # Increment the layer
            isat_info = {
                'description': 'ISAT',
                'folder': self.images_dir,
                'name': os.path.basename(image_path),
                'width': image_width,
                'height': image_height,
                'depth': image.shape[2],  # image depth
                'note': ''
            }
            isat_data = {
                'info': isat_info,
                'objects': isat_objects
            }
            isat_filename = os.path.splitext(os.path.basename(image_path))[0] + '.json'
            isat_file_path = os.path.join(self.output_dir, isat_filename)  # output COCO file path
            with open(isat_file_path, 'w', encoding='utf-8') as file:
                json.dump(isat_data, file, indent=4)

    def from_coco(self):  # 我没处理RLE
        """
        Get the objects information form COCO json file
            Key differences:
            1. 'segmentation'
                COCO: [x1,  y1, x2, y2, ..., xn, yn]
                ISAT: [[x1, y1], [x2, y2], ..., [xn, yn]]
            2. 'bbox'
                COCO: [xmin, ymin, width, height]
                ISAT: [xmin, ymin, xmax, ymax]]
            3. layer
            4. group
        """

        def coco2isat_segmentation(coco_segmentation):
            """Convert COCO segmentation to ISAT segmentation"""
            return [[float(coco_segmentation[idx]), float(coco_segmentation[idx + 1])] for idx in range(0, len(coco_segmentation), 2)]

        def coco2isat(coco_data):
            category_mapping = {category['id']: category['name'] for category in coco_data['categories']}  # map category id to category name
            isat_data_list = []  # to collect information for all images
            for image_info in coco_data['images']:
                isat_data = {
                    'info': {
                        'description': 'ISAT',
                        'folder': '',
                        'name': os.path.splitext(image_info['file_name'])[0],
                        'width': image_info['width'],
                        'height': image_info['height'],
                        'depth': 3,  # assuming RGB
                        'note': ''},
                    'objects': []}
                annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_info['id']]  # check only the given image
                annotations.sort(key=lambda x: -x['area'])  # larger objects first
                group_counter = 1  # group starts from 1
                for annotation in annotations:
                    bbox = [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]]  # xmin, ymin, xmax, ymax
                    group_id = next((obj['group'] for obj in isat_data['objects'] if self.bbox_within(bbox, obj['bbox'])), group_counter)  # group id increases if bbox not within another
                    if group_id == group_counter:
                        group_counter += 1  # new group found
                    layer = sum(obj['group'] == group_id for obj in isat_data['objects']) + 1  # layer inrease by 1
                    isat_object = {
                        'category': category_mapping.get(annotation['category_id'], 'unknown'),  # get the corresponding category name
                        'group': group_id,
                        'segmentation': coco2isat_segmentation(annotation['segmentation'][0]),  # segmentation in [[x1, y1], [x2, y2], ...]
                        'area': annotation['area'],
                        'layer': float(layer),  # 1.0, 2.0, 3.0, ...
                        'bbox': [int(coord) for coord in bbox],  # to integer
                        'iscrowd': annotation['iscrowd'],  # same as in MSCOCO
                        'note': annotation.get('note', '')  # same as in MSCOCO
                    }
                    isat_data['objects'].append(isat_object)  # collect all objects within the given image
                isat_data_list.append(isat_data)  # collect all objects of all images
            return isat_data_list

        json_name = [name for name in os.listdir(self.annotations_dir) if any(name.endswith(file_type) for file_type in ['.json'])][0]  # assuming only one json file
        coco_json_path = os.path.join(self.annotations_dir, json_name)  # get the COCO json file path
        with open(coco_json_path, 'r', encoding='utf-8') as file:
            coco_data = json.load(file)  # load the COCO json file
            isat_datasets = coco2isat(coco_data)  # get ISAT format inforation
            for isat_data in isat_datasets:
                isat_filename = isat_data['info']['name'] + '.json'  # output json file name
                isat_file_path = os.path.join(self.output_dir, isat_filename)  # output COCO file path
                with open(isat_file_path, 'w', encoding='utf-8') as file:
                    json.dump(isat_data, file, indent=4)  # save the json file
