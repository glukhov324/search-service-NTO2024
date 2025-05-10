from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from src.predictor.predictor import Predictor
from src.settings import settings

from PIL import Image
import io

app = FastAPI()

predictor = Predictor(cv_model_wts_path=settings.cv_model_wts_path,
                      ind2name_path=settings.ind2name_decoder_path,
                      ind2cat_path=settings.ind2cat_decoder_path,
                      path_to_base=settings.path_to_base,
                      names_embs_path=settings.names_embs_path)

@app.get('/')
def service_start():
        return "Welcome to my service for NTO. Go to /docs"

@app.post("/predict_cats_names_by_image")
async def image_predict(image_file: UploadFile,
                        city: str = "Нижний Новгород") -> tuple[dict, list[dict]]:
        
        '''
        Возвращает распределение вероятностей для каждой из топ-5 категории, наиболее похожии названия мест и их координаты

        Параметры:
                image_file (UploadFile): входное изображение
                city (str): город, среди достопримечательностей которого нужно осуществлять поиск похожих названий мест

        Возвращаемое значение:
                tuple: кортеж, состоящий из словаря вида {<название_достопримечательности>: вероятность} и 
                списка словарей, каждый из которых имеет ключи Place_name (название достопримечательности), Lon (долгота), Lat (широта)
        '''
        
        if city not in ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"]:
                return JSONResponse(status_code=404, 
                                    content={"message": "Указанного города нет в списке: Екатеринбург, Нижний Новгород, Владимир, Ярославль"})
    

        data = await image_file.read()
        pil_image = Image.open(io.BytesIO(data)).convert('RGB')

        cats_resp, names_resp = predictor.topk_cats_names_by_image(image=pil_image, city=city)

        return (cats_resp, names_resp)


@app.post("/predict_images_by_text")
def text_predict(text: str,
                 city: str = "Нижний Новгород") -> dict:
        
        '''
        Возвращает топ-5 похожих на текстовое описание картинок с координатами найденных мест 

        Параметры:
                text (str): текстовое описание достопримечательности
                city (str): название города, среди достопримечательностей которого нужно осуществлять поиск
        
        Возвращаемое значение:
                dict: словарь для найденных топ-5 мест вида: ключами являются названия похожих мест, содержимое каждого ключа является
                списком из изображения в base64, Lon (долгота), Lat (широта)
        '''
        
        if city not in ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"]:
                return JSONResponse(status_code=404, 
                                    content={"message": "Указанного города нет в списке: Екатеринбург, Нижний Новгород, Владимир, Ярославль"})
        
        names_list = predictor.topk_names_by_text(text=text,
                                                 city=city,
                                                 k=5)
        
        images_names_dict = {name: (predictor.base[predictor.base.Name == name].sample(frac=1).reset_index().iloc[0]['image'],
                                    predictor.base[predictor.base.Name == name].iloc[0]['Lon'],
                                    predictor.base[predictor.base.Name == name].iloc[0]['Lat']) for name in names_list}

        return images_names_dict