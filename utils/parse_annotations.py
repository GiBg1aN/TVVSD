import json
import os
import pandas as pd


def parse_tuhoi(path):
    """
    Read TUHOI annotations and generate captions in the form:
    'subject-verb-object'.

    Args:
        path: TUHOI annotations file path

    Returns:
        A dataframe with columns: 'image_name', 'caption'
    """
    tuhoi_df = pd.read_csv(path, encoding='ISO-8859-1')
    tuhoi_df = tuhoi_df[['image_name', 'tag1', 'orig_tag1']]  # tag1: verb, orig_tag1: object
    tuhoi_df['tag1'] = tuhoi_df['tag1'].apply(lambda x: x.split('\n')[0])  # take first verb only
    tuhoi_df['category_id'] = tuhoi_df.apply(lambda row: row.orig_tag1, 1)
    tuhoi_df['caption'] = tuhoi_df.apply(lambda row: 'person ' + row.tag1 + ' ' + row.orig_tag1, 1)
    tuhoi_df.rename(columns={'image_name': 'image_id'}, inplace=True)
    return tuhoi_df.drop(['tag1', 'orig_tag1'], axis=1)


def parse_coco(path):
    """
    Convert COCO annotations from JSON to DataFrame

    Args:
        path: COCO annotations file path

    Returns:
        A DataFrame with a column for each dictionary key
    """
    fcoco = open(path)
    coco_dictionary = json.load(fcoco)
    return pd.DataFrame(coco_dictionary.get('annotations'))


def parse_coco_categories(object_annotations):
    """
    Convert object annotations from dictionary structure to DataFrame.

    Args:
        object_annotations: a list of dictionaries

    Returns:
        A DataFrame with columns: 'image_id', 'category_id'
    """
    ids = [x.get('image_id') for x in object_annotations]
    categories = [x.get('category_id') for x in object_annotations]
    coco_df = pd.DataFrame()
    coco_df['image_id'] = ids
    coco_df['category_id'] = categories
    return coco_df


def append_row(caption_df, category_df, img_id, id_prefix, new_df):
    """
    Extract the caption and the category related to 'img_id' from
    'caption_df' and append them as a row to 'new_df'.

    Args:
        caption_df: The caption dataset
        category_df: The category dataset
        img_id: The id of image that will be appended
        id_prefix: The prefix the will be prepended to img_id
        new_df: The Dataframe where the row will be appended

    Returns:
        'new_df' with 'img_id' row appended
    """
    row = caption_df[caption_df['image_id'] == img_id]
    if category_df is None:
        category_row = row
    else:
        category_row = category_df[category_df['image_id'] == img_id]

    if id_prefix is not None:
        new_img_id = id_prefix + str(img_id)
    else:
        new_img_id = str(img_id)
    return pd.concat([new_df, pd.DataFrame({'image_id': new_img_id,
                                            'category': category_row.iloc[0].category_id,
                                            'caption': row.caption})])


def main():
    print('Parsing data...')
    caption_train_df = parse_coco('annotations_trainval2014/annotations/captions_train2014.json')
    caption_val_df = parse_coco('annotations_trainval2014/annotations/captions_val2014.json')
    object_train_df = parse_coco('annotations_trainval2014/annotations/instances_train2014.json')
    object_val_df = parse_coco('annotations_trainval2014/annotations/instances_val2014.json')

    categories_labels = [line.rstrip('\n') for line in open('coco-labels-paper.txt')]
    categories_labels = {str(i + 1): categories_labels[i] for i in range(len(categories_labels))}

    object_train_df['category_id'] = object_train_df['category_id'].apply(
        lambda id: categories_labels.get(str(id)))
    object_val_df['category_id'] = object_val_df['category_id'].apply(
        lambda id: categories_labels.get(str(id)))

    tuhoi_df = parse_tuhoi('crowdflower_result.csv')

    # Get annotations only for downloaded images
    img_list = [os.path.splitext(x)[0] for x in os.listdir('dataset')]

    new_df1 = pd.DataFrame(columns=['image_id', 'category', 'caption'])
    new_df2 = pd.DataFrame(columns=['image_id', 'category', 'caption'])
    new_df3 = pd.DataFrame(columns=['image_id', 'category', 'caption'])

    print('Aggregating...')
    for img_name in img_list:
        if img_name.startswith('COCO_train2014_'):
            img_id = int(img_name[len('COCO_train2014_'):])
            new_df1 = append_row(caption_train_df, object_train_df, img_id,
                                 'COCO_train2014_', new_df1)

        elif img_name.startswith('COCO_val2014_'):
            img_id = int(img_name[len('COCO_val2014_'):])
            new_df2 = append_row(caption_val_df, object_val_df, img_id, 'COCO_val2014_', new_df2)
        else:
            new_df3 = append_row(tuhoi_df, None, img_name, None, new_df3)

    new_df = pd.concat([new_df1, new_df2, new_df3])
    new_df.reset_index(drop=True, inplace=True)

    print(len(new_df.image_id), 'rows.')
    print(len(new_df.image_id.unique()), 'unique annotations.')
    print('Writing...')
    new_df.to_csv('filtered_annotations.csv', index=False)


if __name__ == '__main__':
    main()
