import os

def download_coco_part(file_path='coco_dataset/val2017.zip',extract_to='coco_dataset/val2017'):
    import os
    import requests
    import zipfile

    # 创建目录以存储数据集
    os.makedirs("coco_dataset", exist_ok=True)

    urls = [
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ]
    
    for url in urls:
        response = requests.get(url)
        with open(os.path.join("coco_dataset", os.path.basename(url)), 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {url}")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path} to {extract_to}")

if __name__ == "__main__":
    download_coco_part()

    
