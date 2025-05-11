import pandas as pd
import numpy as np
import os
import torch
import pickle
import faiss
import PIL
from sentence_transformers import SentenceTransformer

from src.cv_model import Model, data_transforms
from src.settings import settings
from src.names_embeddings import get_names_embeddings
from src.schemas import SearchByImageResult, SearchByTextResult




class Predictor:
    def __init__(self,
                 cv_model_wts_path: str,
                 ind2name_decoder_path: str,
                 ind2cat_decoder_path: str,
                 path_to_search_base: str,
                 names_embs_path: str):
        
        with open(f"{ind2name_decoder_path}", "rb") as fp:
            self.ind2name = pickle.load(fp)

        with open(f"{ind2cat_decoder_path}", "rb") as fp:
            self.ind2cat = pickle.load(fp)

        self.search_base = pd.read_csv(path_to_search_base)

        self.cv_model = Model(num_cats=len(self.ind2cat),
                              num_names=len(self.ind2name)).to(settings.DEVICE)
        self.cv_model.load_state_dict(torch.load(f"{cv_model_wts_path}"))
        self.cv_model.eval()

        self.rubert = SentenceTransformer(settings.EMBEDDINGS_MODEL_NAME,
                                          device=settings.DEVICE)

        if not os.path.exists(names_embs_path):
            get_names_embeddings(search_base=self.search_base,
                                 model=self.rubert)
            
        names, cities, embs = np.load(names_embs_path, allow_pickle=True)
        self.name_emb_base = pd.DataFrame({
                "Name": names,
                "City": cities,
                "Name_embedding": embs
            })


    @torch.inference_mode()
    def topk_cats_names_by_image(self,
                                 image: PIL.Image,
                                 city: str,
                                 k: int = settings.TOPK) -> SearchByImageResult:
        '''
        Нахождение по входящему изображению распределения вероятностей на top-k категорий достопримечательностей, 
        top-k названий наиболее похожих достопримечательностей с их координатами

        :param image (PIL.Image): входящее изображение
        :param city (str): город для поиска похожих названий достопримечательностей
        :param k (int): ограничение выдачи распределения вероятностей категорий и количества названий похожих достопримечательностей

        :return: 
            распределение вероятностей на top-k категорий достопримечательностей, top-k названий наиболее похожих достопримечательностей с их координатами
        '''
        
        temp_base = self.name_emb_base[self.name_emb_base.City == city]
        c_pl_names = temp_base["Name"].unique().tolist()
        
        processed_image = data_transforms(image).unsqueeze(0).to(settings.DEVICE)
        cats_logits, names_logits = self.cv_model(processed_image)

        out_topk_cats = cats_logits.topk(k=k)
        out_topk_names = names_logits.topk(k=len(self.ind2name))

        cats_probs = out_topk_cats.values.softmax(dim=-1).cpu().tolist()[0]
        cats = [self.ind2cat[int(i)] for i in out_topk_cats.indices.cpu()[0]]
        cats_probs_resp = [{"category": cats[i],
                            "probability": cats_probs[i]} for i in range(len(cats))]

        names = [self.ind2name[int(i)] for i in out_topk_names.indices.cpu()[0] if self.ind2name[int(i)] in c_pl_names][:k]
        names_coords_resp = [
            {
                "sight_name": name, 
                "lon": self.search_base[self.search_base.Name == name]['Lon'].values[0], 
                "lat": self.search_base[self.search_base.Name == name]['Lat'].values[0]
            } for name in names
            ]
        
        response = {
            "cats_probs": cats_probs_resp,
            "names_coords": names_coords_resp
        }
    

        return response
    
    pd.DataFrame({

    })

    def topk_images_by_text(self,
                            text: str,
                            city: str,
                            k: int = settings.TOPK) -> SearchByTextResult:
        '''
        Нахождение top-k наиболее соответствующих изображений достопримечательностей на входящий текстовый запрос

        :param text (str): входящий текстовый запрос
        :param city (str): город для поиска похожих изображений достопримечательностей на входящий текстовый запрос
        :param k (int): ограничение выдачи количества похожих изображений достопримечательностей на входящий текстовый запрос
        
        :return:
            top-k наиболее похожих изображений достопримечательностей для входящего текстового запроса
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
        names_list = pd.merge(results, temp_base, left_on='ann', right_index=True)[:k]["Name"].tolist()
    
        images_names_dict = [
            {
                "sight_name": name,
                "lon": self.search_base[self.search_base.Name == name].iloc[0]['Lon'],
                "lat": self.search_base[self.search_base.Name == name].iloc[0]['Lat'],
                "encoded_img": self.search_base[self.search_base.Name == name].sample(frac=1).reset_index().iloc[0]['image']
            } for name in names_list
        ]
        response = {
            "result": images_names_dict
        }
        return response



predictor = Predictor(cv_model_wts_path=settings.CV_MODEL_WTS_PATH,
                      ind2name_decoder_path=settings.IND2NAME_DECODER_PATH,
                      ind2cat_decoder_path=settings.IND2CAT_DECODER_PATH,
                      path_to_search_base=settings.SEARCH_BASE_PATH,
                      names_embs_path=settings.NAMES_EMBEDDINGS_PATH)