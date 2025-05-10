import os
import logging
import torch
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env")
    
    HOST: str = "0.0.0.0"
    PORT: int = "8000"
    DEV_MODE: bool = True
    TITLE: str = "NTO SEARCH SERVICE"

    # Paths settings
    dataset_name: str ='dataset'
    dataset_main_folder: str = './dataset_prepare'
    df_path: str = 'data'
    del_imgs_path: str ='/plots_deleted_images'

    # names embeddings settings
    path_to_base: str = 'src/predictor/train_val.csv'
    names_embs_path: str = 'src/predictor/names_embeddings.npy'

    # cv model
    cv_model_wts_path: str = 'src/cv_model/best_wts_cat_names.pt'
    ind2name_decoder_path: str = 'src/cv_model/ind2name.pkl'
    ind2cat_decoder_path: str = 'src/cv_model/ind2cat.pkl'

    # Logging settings
    LOG_FORMAT: str = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
    LOG_FOLDER: str = os.path.join("logs")
    LOG_FILE: str = "logs.log"
    LOG_LEVEL: int = logging.INFO

    DEVICE: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    USE_CITIES: List[str] = ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"]
    DEFAULT_CITY: str = "Нижний Новгород"


settings = Settings()