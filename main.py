import numpy as np
import pandas as pd
import random
import kagglehub

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import os
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import gc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)

images_path = '/kaggle/input/flickr8k/Images/'
captions_data_path = '/kaggle/input/flickr8k/captions.txt'

# Загрузка данных
data = pd.read_csv(captions_data_path)
image_ids = data['image'].values
captions_list = data['caption'].values

data.head(3)
data.tail(2)
data.info()

data_idx = random.randint(0, len(data)-1)

image = images_path + data.iloc[data_idx,0]
rnd_img = mpimg.imread(image)
plt.imshow(rnd_img)
plt.show()

for i in range(data_idx, data_idx+1):
    print('\n', "Caption:", data.iloc[i,1])

# Извлечение признаков изображений
# Загрузка модели ResNet50
base_model = ResNet50(weights='imagenet')
model = Model(base_model.input, base_model.layers[-2].output)

def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))  # Изменение размера на 224x224 для ResNet
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

# Лимит на количество изображений
limit = 1000

# Извлечение признаков изображений с лимитом
image_features = np.zeros((min(len(image_ids), limit), 2048))
for i, img_id in enumerate(image_ids[:limit]):
    img_path = os.path.join(images_path, img_id)
    image_features[i] = extract_features(img_path, model)

# Токенизация подписей
max_vocab_size = 5000
caption_tokenizer = Tokenizer(num_words=max_vocab_size)
caption_tokenizer.fit_on_texts(captions_list)
caption_sequences = caption_tokenizer.texts_to_sequences(captions_list)
caption_word_index = caption_tokenizer.word_index
max_len_caption = 30
decoder_input_data = pad_sequences(caption_sequences, maxlen=max_len_caption, padding='post')

# Подготовка данных для обучения
num_decoder_tokens = min(max_vocab_size, len(caption_word_index) + 1)
decoder_target_data = np.zeros((len(captions_list), max_len_caption, num_decoder_tokens), dtype='float32')
for i, seqs in enumerate(caption_sequences):
    for t, token in enumerate(seqs):
        if t < max_len_caption - 1:  # Убедитесь, что не выходите за пределы
            decoder_target_data[i, t, token] = 1.0

# Гиперпараметры
latent_dim = 256

# Модель Seq2Seq
image_input = Input(shape=(2048,))
image_dense = Dense(latent_dim, activation='relu')(image_input)  # Преобразование размерности

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[image_dense, image_dense])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([image_input, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
history = model.fit([image_features, decoder_input_data], decoder_target_data,
                    batch_size=64, epochs=30, validation_split=0.2)

# Построение графика потерь
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Выбираем случайный индекс
random_index = random.randint(0, len(image_ids) - 1)

# Получаем путь к изображению и соответствующую подпись
random_image_id = image_ids[random_index]
random_caption = captions_list[random_index]

# Загружаем и отображаем изображение
img_path = os.path.join(images_path, random_image_id)
img = load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.axis('off')  # Отключаем оси
plt.text(0.5, -0.1, random_caption, ha='center', va='center', transform=plt.gca().transAxes)

plt.show()
