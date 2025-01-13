from src.cv_model.model import Model
from src.cv_model.data_transforms import data_transforms

import torch
import pickle
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from loguru import logger


class Predictor:
    def __init__(self,
                 cv_model_wts_path: str,
                 ind2name_path: str,
                 ind2cat_path: str,
                 path_to_base: str,
                 names_embs_path: str):
        
        with open(f"{ind2name_path}", 'rb') as fp:
            self.ind2name = pickle.load(fp)

        with open(f"{ind2cat_path}", 'rb') as fp:
            self.ind2cat = pickle.load(fp)

        self.base = pd.read_csv(path_to_base)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.cv_model = Model(num_cats=len(self.ind2cat),
                              num_names=len(self.ind2name)).to(self.device)
        self.cv_model.load_state_dict(torch.load(f'{cv_model_wts_path}'))
        self.cv_model.eval()

        self.rubert = SentenceTransformer('cointegrated/rubert-tiny2')

        try:
            names, cities, embs = np.load(names_embs_path, allow_pickle=True)
            self.name_emb_base = pd.DataFrame({
                "Name": names,
                "City": cities,
                "Name_embedding": embs
            })
        except:
            raise Exception("Base of names and names' embeddings does not exist or incorrect. Please, run src/names_embs_prepare/embeddings_prepare.py")

    @torch.inference_mode()
    def topk_cats_names_by_image(self,
                                 image: Image,
                                 k: int = 5) -> dict:
        
        '''
        Возвращает логиты и индексы для топ-k категорий, для названий возвращает логиты и индексы в порядке убывания веротностей

        Параметры:
            image (PIL.Image): изображение, отправленное пользователем
            k (int): количество наиболее подходящих категорий изображения для выдачи

        Возвращаемое значение:
            словарь длины k вида <категория достопримечательности: вероятность того, что категория подходит изображению пользователя>
        '''
        
        processed_image = data_transforms(image).unsqueeze(0).to(self.device)
        cats_logits, names_logits = self.cv_model(processed_image)

        out_topk_cats = cats_logits.topk(k=k)
        out_topk_names = names_logits.topk(k=len(self.ind2name))

        return {
            "topk_cats_logits": out_topk_cats.values.cpu(),
            "topk_cats_indices": out_topk_cats.indices.cpu(),
            "topk_names_logits": out_topk_names.values.cpu(),
            "topk_names_indices": out_topk_names.indices.cpu()
        }
    

    def topk_names_by_text(self,
                           text: str,
                           city: str,
                           k: int = 5) -> list:
        
        '''
        Возвращает k наиболее похожих названий на входящий текстовый запрос

        Параметры:
            text (str): текстовый запрос
            city (str): город для поиска похожих достопримечательностей
            k (int): количество наиболее похожих достопримечательностей для выдачи
        
        Возвращаемое значение:
            список из k наиболее подходящих названий для входящего текстового запроса
        '''
        
        temp_base = self.name_emb_base[self.name_emb_base.City == city]
        temp_base = temp_base.reset_index()
        vectors = np.array([vec for vec in temp_base["Name_embedding"]])

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        _vector = np.array([self.rubert.encode(text)])
        faiss.normalize_L2(_vector)

        distances, ann = index.search(_vector, k=index.ntotal)
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        merged = pd.merge(results, temp_base, left_on='ann', right_index=True)

        return merged[:k]["Name"].tolist()