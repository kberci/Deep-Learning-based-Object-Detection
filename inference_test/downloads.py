from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path
import pandas as pd
import requests
import os


def download_images(csv_path, save_path):
    d = pd.read_csv(csv_path, squeeze=True)
    dict_list = [{d.columns[i]: item[i] for i in range(len(d.columns))} for item in d.values]
    for img in dict_list:
        img_path = save_path / img['name']
        if not img_path.is_file():
            resp = requests.get(img['url'], allow_redirects=True)
            print("Saving: ", img_path)
            open(img_path, 'wb').write(resp.content)
        else:
            print("(Image " + img_path + " already exists, skip download.)")


def download_models(csv_path, save_path):
    d = pd.read_csv(csv_path, squeeze=True)
    dict_list = [{d.columns[i]: item[i] for i in range(len(d.columns))} for item in d.values]
    for model in dict_list:
        config_path = save_path / model['name']
        if not config_path.is_file():
            gdd.download_file_from_google_drive(file_id=model['id'].split('/')[-2], dest_path=config_path)
        else:
            print("Model was already downloaded: ", config_path)


PATH_TO_CONFIGS = Path('config')
PATH_TO_IMAGES = Path('images')
PATH_TO_MODELS = Path('models')

# Create directories if they do not exist and download images
if not PATH_TO_IMAGES.is_dir():
    os.mkdir(PATH_TO_IMAGES)
    download_images(PATH_TO_CONFIGS / 'images.csv', PATH_TO_IMAGES)
else:
    print("Image folder already exists, skip downloading images.")

# Create directories if they do not exist and download models
model_folders = ['edgetpu', 'mobilenet', 'mobiledet', 'yolo4', 'yolo5', 'dnn_files']
if not PATH_TO_MODELS.is_dir():
    os.mkdir(PATH_TO_MODELS)
else:
    print("Model folder already exists.")
for model_folder in model_folders[:1]:
    if not (PATH_TO_MODELS / model_folder).is_dir():
        os.mkdir(PATH_TO_MODELS / model_folder)
    else:
        print("Sub-model folder already exists.")
    download_models((PATH_TO_CONFIGS / model_folder).with_suffix('.csv'), PATH_TO_MODELS / model_folder)
