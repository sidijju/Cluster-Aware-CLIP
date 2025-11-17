set -e

mkdir data
mkdir data/coco
cd data/coco

echo "Downloading and unzipping train2017"
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

echo "Downloading and unzipping annotations_trainval2017"
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

echo "Downloading and unzipping val2017"
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip