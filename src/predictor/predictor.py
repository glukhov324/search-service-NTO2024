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
            raise Exception("Отсутствуют подготовленные эмбеддинги названий достопримечательностей. Пожалуйста, запустите скрипт src/names_embs_prepare/embeddings_prepare.py")

    @torch.inference_mode()
    def topk_cats_names_by_image(self,
                                 image: Image,
                                 city: str,
                                 k: int = 5) -> tuple[dict, list[dict]]:
        
        '''
        Функция находит распределение вероятностей для топ-k категорий достопримечательностий, топ-5 названий мест с координатами

        Параметры: 
            image (PIL.Image): изображение, отправленное пользователем
            city (str): город, в котором нужно искать достопримечательность
            k (int): количество наиболее подходящих категорий изображения для выдачи

        Возвращаемое значение:
            tuple: кортеж, состоящий из словаря вида {<название_достопримечательности>: вероятность} и 
            списка словарей, каждый из которых имеет ключи Place_name (название достопримечательности), Lon (долгота), Lat (широта)
        '''
        
        temp_base = self.name_emb_base[self.name_emb_base.City == city]
        c_pl_names = temp_base["Name"].unique().tolist()
        
        processed_image = data_transforms(image).unsqueeze(0).to(self.device)
        cats_logits, names_logits = self.cv_model(processed_image)

        out_topk_cats = cats_logits.topk(k=k)
        out_topk_names = names_logits.topk(k=len(self.ind2name))

        cats_probs = out_topk_cats.values.softmax(dim=-1).cpu().tolist()[0]
        cats = [self.ind2cat[int(i)] for i in out_topk_cats.indices.cpu()[0]]
        cats_resp = {cats[i]: cats_probs[i] for i in range(len(cats))}

        names = [self.ind2name[int(i)] for i in out_topk_names.indices.cpu()[0] if self.ind2name[int(i)] in c_pl_names][:k]
        names_resp = [{"Place_name": name, "Lon": self.base[self.base.Name == name]['Lon'].values[0], "Lat": self.base[self.base.Name == name]['Lat'].values[0]} for name in names]

        return (cats_resp, names_resp)
    

    def topk_names_by_text(self,
                           text: str,
                           city: str,
                           k: int = 5) -> list:
        
        '''
        Функция находит k наиболее похожих названий на входящий текстовый запрос

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

        vector = np.array([self.rubert.encode(text)])
        faiss.normalize_L2(vector)

        distances, ann = index.search(vector, k=index.ntotal)
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        merged = pd.merge(results, temp_base, left_on='ann', right_index=True)

        return merged[:k]["Name"].tolist()