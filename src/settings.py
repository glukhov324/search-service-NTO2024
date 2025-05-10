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
    PROCESSED_DATASET_NAME: str ='dataset'
    DATASET_MAIN_FOLDER: str = './dataset_prepare'
    RAW_DATA_PATH: str = 'data'
    DEL_IMGS_PATH: str ='/plots_deleted_images'
    DELETE_IMAGE_THRESHOLD: float = 0.9

    # names embeddings settings
    EMBEDDINGS_MODEL_NAME: str = "cointegrated/rubert-tiny2"
    SEARCH_BASE_PATH: str = 'src/predictor/train_val.csv'
    NAMES_EMBEDDINGS_PATH: str = 'src/predictor/names_embeddings.npy'

    # cv model
    CV_MODEL_WTS_PATH: str = 'src/cv_model/best_wts_cat_names.pt'
    IND2NAME_DECODER_PATH: str = 'src/cv_model/ind2name.pkl'
    IND2CAT_DECODER_PATH: str = 'src/cv_model/ind2cat.pkl'

    # Logging settings
    LOG_FORMAT: str = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
    LOG_FOLDER: str = os.path.join("logs")
    LOG_FILE: str = "logs.log"
    LOG_LEVEL: int = logging.INFO

    DEVICE: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    USE_CITIES: List[str] = ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"]
    DEFAULT_CITY: str = "Нижний Новгород"


settings = Settings()