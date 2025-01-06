# НТО 2024: поиск достопримечательностей

Данные предоставлены для Екатеринбурга, Нижнего Новгорода, Владимира, Ярославля

Исходный датасет был соединён в один из датасетов для Екатеринбурга, Нижнего Новгорода, Владимира, Ярославля и частично очищен от чёрно-белых изображений и изображений-коллажей при помощи модели ruclip 

### Какие задачи решает сервис?

1. Определение по изображению топ-5 наиболее подходящих категорий и топ-5 наиболее подходящих названий
2. Определение по текстовому описанию топ-5 наиболее подходящих изображений мест

Для 1 пункта используется классификатор с двумя головами (одна голова определяет категорию места, другая голова определяет название места),
для извлечения признаков из изображения была выбрана архитектура MobileNet. Всего:  25 категорий и 387 названий мест

Целевая метрика соревнования: precision@5

Полученная модель имеет precision@5 = 0.76 для названий мест и precision@5 = 0.94 для категорий


Для пункта 2 используется модель ruberttiny2 для извлечения эмбеддингов из названий мест, применяется библиотека faiss 
для ранжирования названий из датасета относительно входящего текстового описания достопримечательности, для топ-5 наиболее подходящих названий из датасета
выдаются изображения, соответствующие этим названиям




python version is 3.10

__installation guide__:

windows:
1. py -3.10 -m venv venv
2. .\venv\Scripts\activate

linux/macOS
1. python3.10 -m venv venv
2. source venv/bin/activate

python -m pip install pip==23.0.1

pip install cython==3.0.11

pip install -r .\requirements.txt

pip install transformers==4.47.1

pip install huggingface_hub==0.25.0

pip install sentence_transformers==3.3.1

$env:PYTHONPATH = $pwd  - windows
export PYTHONPATH=$PYTHONPATH:$PWD   - linux/macOS


Скачайте все папки из https://drive.google.com/drive/folders/1NWGmIqbgzb2MQ9ad3PBifPaqbAaSR598?usp=sharing

1. Положите файлы из скачанной папки predictor в src/predictor   
2. Положите файлы из скачанной папки cv_model в src/cv_model
3. Переместите скачанную папку data в корень проекта
4. Положите файлы из скачанной папки cv_model_train to src/cv_model_train

3 и 4 пункты необязательны

3 - для дальнейшего создания единого датасета из исходных данных

4 - для обучения cv-модели



__Start guide__:

windows:
1. .\venv\Scripts\activate
2. $env:PYTHONPATH = $pwd

linux/macOS:
1. source venv/bin/activate
2. export PYTHONPATH=$PYTHONPATH:$PWD

3. python -m uvicorn app:app --reload

В demo_test_service.ipynb находится демонстрация работы сервиса