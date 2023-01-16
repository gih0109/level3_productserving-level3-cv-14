import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Merge annotation json files')
    parser.add_argument('src_file0', help='json file path') # 합치고 싶은 json file path
    parser.add_argument('src_file1', help='json file path') # 합치고 싶은 json file path
    parser.add_argument('dst_file', help='new json file path') # 만들어질 json file path
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open(args.src_file0, 'r') as f:
        json0 = json.load(f)
    with open(args.src_file1, 'r') as f:
        json1 = json.load(f)
    
    for i in range(len(json0['images'])):
        json0['images'][i]['id']
    for i in range(len(json0['annotations'])):
        json0['annotations'][i]['image_id']
    
    for i in range(len(json1['images'])):
        json1['images'][i]['id'] += len(json0['images'])
    for i in range(len(json1['annotations'])):
        json1['annotations'][i]['image_id'] += len(json0['images'])
    
    info = json0['info']
    licenses = json0['licenses']
    categories = json0['categories']
    images = []
    annotations = []

    for x in json0['images']:
        images.append(x)
    for x in json1['images']:
        images.append(x)

    for x in json0['annotations']:
        annotations.append(x)
    for x in json1['annotations']:
        annotations.append(x)

    for i in range(len(annotations)):
        annotations[i]['id'] = i
    
    json_new = dict()
    json_new['info'] = info
    json_new['licenses'] = licenses
    json_new['categories'] = categories
    json_new['images'] = images
    json_new['annotations'] = annotations

    with open(args.dst_file, 'w') as f:
        json.dump(json_new, f)


if __name__ == '__main__':
    main()