pip install --upgrade torch
pip install loguru

python -c "import nltk; nltk.download('stopwords')"

# Uncomment the following to download the dataset from kaggle
# Need to upload your api key from kaggle.json file
# Instructions from: https://medium.com/@opalkabert/downloading-kaggle-datasets-into-google-colab-fb9654c94235

# pip install -U -q kaggle

# mkdir -p ~/.kaggle
# cp kaggle.json ~/.kaggle/
# chmod 600 /root/.kaggle/kaggle.json

# kaggle datasets download -d datasnaek/youtube-new
# unzip youtube-new.zip