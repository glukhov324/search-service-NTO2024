import pandas as pd
import torch
import clip
from tqdm import tqdm
import PIL
import io
import os
import base64
import matplotlib.pyplot as plt

from src.logger import logger
from src.settings import settings



def base64_to_pil_image(image_base64: str) -> PIL.Image:
    """
    Конвертация изображения, закодированного в base64, в PIL.Image
    """

    return PIL.Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8")))).convert('RGB')


class DatasetPrepare:
    def __init__(self, 
                 df_path: str = 'data',
                 del_imgs_pth: str = 'plots_deleted_images',
                 dataset_name: str = 'dataset',
                 dataset_main_folder: str = 'dataset_prepare'):
        
        self.dataset_main_folder = dataset_main_folder
        self.df_path = df_path
        self.del_imgs_pth = del_imgs_pth
        self.dataset_name = dataset_name
        self.dataset = pd.DataFrame()
        self.model_clip, self.processor = clip.load("ViT-B/32", device=settings.DEVICE)
    


    def create_dataset(self) -> None:
        """
        Создание датасета для городов: Екатеринбург, Нижний Новгород, Владимир, Ярославль
        """

        self.dataset = pd.DataFrame()

        logger.info("Creating dataset was started")

        for city in tqdm(["EKB", "NN", "Vladimir", "Yaroslavl"]):

            df_images = pd.read_csv(f"{self.df_path}/{city}_images.csv")
            df_images = df_images.rename(columns={"name": "Name", "img": "image"})

            df_places = pd.read_csv(f"{self.df_path}/{city}_places.csv")
            df_temp = df_images.merge(df_places, left_on="Name", right_on="Name")

            self.dataset = pd.concat((self.dataset, df_temp), axis=0)

        self.dataset = self.dataset.drop_duplicates()
        self.dataset = self.dataset.reset_index(drop=True)

        logger.info("Dataset was created!")



    def _save_figure_deleted_images(self, score_img: dict) -> None:
        """
        Сохранение восьми сэмплов удалённых изображений из датасета, имеющих наибольшую вероятность соответствия ключевому запросу для удаления

        :param score_img (dict): словарь, в котором изображению соотносится вероятность соответствия ключевому запросу для удаления
        """

        if not os.path.exists(f"{self.del_imgs_pth}"):
            os.mkdir(f"{self.del_imgs_pth}")

        short_imgs_list = [t[1][1] for t in sorted(score_img.items(), key=lambda x:(x[1][0]), reverse=True)][:8]

        plt.figure(figsize=(16, 8))

        if len(short_imgs_list) >= 8:
            name = len(os.listdir(f"{self.del_imgs_pth}"))
            for i in range(4):
                plt.subplot(2, 4, i+1)
                plt.imshow(short_imgs_list[i])
                plt.subplot(2, 4, i+5)
                plt.imshow(short_imgs_list[i+4])
            plt.savefig(f"{self.del_imgs_pth}/plot_{name}.png")


    @torch.inference_mode()
    def clean_dataset(self,
                      texts: list[str],
                      threshold: float = 0.9) -> None:
        """
        Очистка датасета при помощи модели clip. 
        Удаление строк из датасета, если изображения в этих строках похожи с текстом ключевого запроса для очистки датасета
        в списке texts более, чем на threshold

        texts[0] - ключевой запрос для очистки датасета

        :param texts (list[str]): список запросов
        :param threshold (float): порог удаления строки из датасета по нулевому запросу из texts
        """
        
        ids_to_delete = []
        score_img = {}

        logger.info(f"Dataset cleaning was started by query: {texts[0]}")

        for i in tqdm(range(len(self.dataset))):

            pil_img = base64_to_pil_image(self.dataset.iloc[i]['image'])
            prepoc_img = self.processor(pil_img).unsqueeze(0).to(settings.DEVICE)
            tokenized_texts = clip.tokenize(texts).to(settings.DEVICE)

            logits_per_image, _ = self.model_clip(prepoc_img, tokenized_texts)
            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

            if probs[0][0] > threshold:
                ids_to_delete.append(i)
                score_img[i] = (probs[0][0], pil_img)
        
        self._save_figure_deleted_images(score_img=score_img) 

        self.dataset = self.dataset.drop(ids_to_delete, axis=0)
        self.dataset = self.dataset.reset_index(drop=True)

        logger.info(f"Dataset cleaning was ended by query: {texts[0]}")


    def save_dataset(self) -> None:
        """
        Сохранение текущей версии датасета
        """

        self.dataset.to_csv(f"{self.dataset_main_folder}/{self.dataset_name}", index=False)
    
dp = DatasetPrepare(df_path=settings.RAW_DATA_PATH,
                    del_imgs_pth=settings.DEL_IMGS_PATH,
                    dataset_name=settings.PROCESSED_DATASET_NAME,
                    dataset_main_folder=settings.DATASET_MAIN_FOLDER)