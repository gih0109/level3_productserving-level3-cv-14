import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')

    parser.add_argument('image_version', help='version of images(e.g. train_v1-1)')
    parser.add_argument('json_version', help='version of json file(e.g. train_v1-1)')
    parser.add_argument('years_train', help='years of train set(e.g. 2013,2014,2015)')
    # parser.add_argument('years_val', help='years of val set(e.g. 2016,2017,2018)')
    parser.add_argument('name_train', help='name of the new train json file(e.g. new_train)')
    parser.add_argument('name_val', help='name of the new val json file(e.g. new_val)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    image_path = '/opt/ml/input/data/images/'
    json_path = '/opt/ml/input/data/annotations/'

    train = list(args.years_train.split(','))
    # val = list(args.years_val.split(','))

    # image 옮기기
    imgs = os.listdir(image_path + args.image_version)
    for x in imgs:
        if x[:4] in train:
            if not os.path.isdir(image_path + args.name_train):
                os.mkdir(image_path + args.name_train)
            os.rename(image_path + f'{args.image_version}/{x}', image_path + f'{args.name_train}/{x}')
        else:
            if not os.path.isdir(image_path + args.name_val):
                os.mkdir(image_path + args.name_val)
            os.rename(image_path + f'{args.image_version}/{x}', image_path + f'{args.name_val}/{x}')
    
    # json 쪼개기
    with open(json_path + f'{args.json_version}.json') as f:
        data = json.load(f)
    info = data['info']
    licenses = data['licenses']
    categories = data['categories']
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []

    train_id = []
    for x in data['images']:
        if x['file_name'][:4] in train:
            train_images.append(x)
            train_id.append(x['id'])
        else:
            val_images.append(x)
    for x in data['annotations']:
        if x['image_id'] in train_id:
            train_annotations.append(x)
        else:
            val_annotations.append(x)
    
    train_json = dict()
    val_json = dict()

    train_json['info'] = info
    train_json['licenses'] = licenses
    train_json['categories'] = categories
    train_json['images'] = train_images
    train_json['annotations'] = train_annotations
    val_json['info'] = info
    val_json['licenses'] = licenses
    val_json['categories'] = categories
    val_json['images'] = val_images
    val_json['annotations'] = val_annotations

    with open(json_path + f'{args.name_train}.json', 'w') as f:
        json.dump(train_json, f)
    with open(json_path + f'{args.name_val}.json', 'w') as f:
        json.dump(val_json, f)


if __name__ == '__main__':
    main()