from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from config.config import Config
from loguru import logger


def encode_names(names: list[str],
                 cities: list[str],
                 model: SentenceTransformer) -> pd.DataFrame:
    
    '''
    Получение эмбеддингов названий мест

    Параметры:
        names (list): список уникальных названий мест
        cities (list): список городов, относящихся к названиям мест в names. i-ый город соответствует i-му названию места
        model (SentenceTransformer): модель, с помощью которой будет получать эмбеддинги из названий мест
    
    Возвращаемое значение:
        pd.DataFrame из names, cities, name_embeddings (эмбеддингов названий)
    '''

    name_embeddings = []

    for i in tqdm(range(len(names))):
        name_embeddings.append(model.encode(names[i]))

    return pd.DataFrame({
        "Name": names,
        "City": cities,
        "Name_embedding": name_embeddings
    })


if __name__ == "__main__":

    ruberttiny2 = SentenceTransformer('cointegrated/rubert-tiny2')
    df_base = pd.read_csv(Config.path_to_base)
    names = df_base["Name"].unique().tolist() 
    cities = [df_base[df_base.Name == name]['City'].values[0] for name in names]

    df_embs = encode_names(names=names,
                           cities=cities,
                           model=ruberttiny2)
    
    np.save(Config.names_embs_path, (df_embs["Name"].values, 
                                     df_embs["City"].values, 
                                     df_embs["Name_embedding"].values))

    logger.info("Base of names' embeddings was created!")