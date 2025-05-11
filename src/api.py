from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from src.settings import settings
from src.predictor import main_predictor
from src.logger import logger
from src.schemas import SearchByImageResult, SearchByTextResult


router = APIRouter(prefix="/search")
responses = {
    200: {"description": 'Success'},
    204: {"description": 'Not found valid content in request'},
    404: {"description": 'Bad request'},
    501: {"description": 'Not implemented'},
    415: {"description": 'Unsupported media type in request'}
}


@router.post("/by_image", tags=["SEARCH"], response_model=SearchByImageResult, responses={**responses})
async def cats_sim_names_by_img(image_file: UploadFile,
                                city: str = settings.DEFAULT_CITY):
        
        if city not in settings.USE_CITIES:
                return JSONResponse(status_code=404, 
                                    content={"message": f"Указанного города нет в списке: {settings.USE_CITIES}"})
    

        data = await image_file.read()
        pil_image = Image.open(io.BytesIO(data)).convert('RGB')

        logger.info("Start predicting categories of input sight image and names of similar sights")

        response = main_predictor.topk_cats_names_by_image(image=pil_image, 
                                                           city=city)
        logger.info("End predicting categories of input sight image and names of similar sights")

        return response


@router.post("/by_text", tags=["SEARCH"], response_model=SearchByTextResult, responses={**responses})
async def sim_imgs_by_text(text: str,
                       city: str = settings.DEFAULT_CITY):
        
        if city not in settings.USE_CITIES:
                return JSONResponse(status_code=404, 
                                content={"message": f"Указанного города нет в списке: {settings.USE_CITIES}"})
        
        logger.info("Start searching images of similar sights")

        images_names_dict = main_predictor.topk_images_by_text(text=text,
                                                               city=city)

        logger.info("End searching images of similar sights")

        return images_names_dict