from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.settings import settings
from src.logger.main_logger import logger


def get_names_embeddings(search_base: pd.DataFrame,
                         model: SentenceTransformer) -> pd.DataFrame:
    
    '''
    Получение эмбеддингов названий достопримечательностей по базе для поиска

    :param search_base (pd.DataFrame): базе для поиска достопримечательностей
    :param model (SentenceTransformer): модель для получения эмбеддингов названий достопримечательностей
    '''

    names = search_base["Name"].unique().tolist() 
    cities = [search_base[search_base.Name == name]['City'].values[0] for name in names]
    names_embeddings = []

    for i in tqdm(range(len(names))):
        names_embeddings.append(model.encode(names[i]))

    np.save(file=settings.NAMES_EMBEDDINGS_PATH, 
            arr=(names, cities, names_embeddings))

    logger.info("Base of names embeddings was created!") 