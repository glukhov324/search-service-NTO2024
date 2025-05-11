from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from src.settings import settings
from src.predictor import main_predictor


router = APIRouter(prefix="/search")


@router.post("/by_image")
async def image_predict(image_file: UploadFile,
                        city: str = settings.DEFAULT_CITY) -> tuple[dict, list[dict]]:
        
        if city not in settings.USE_CITIES:
                return JSONResponse(status_code=404, 
                                    content={"message": f"Указанного города нет в списке: {settings.USE_CITIES}"})
    

        data = await image_file.read()
        pil_image = Image.open(io.BytesIO(data)).convert('RGB')

        cats_resp, names_coords_resp = main_predictor.topk_cats_names_by_image(image=pil_image, 
                                                                               city=city)

        return (cats_resp, names_coords_resp)


@router.post("/by_text")
def text_predict(text: str,
                 city: str = settings.DEFAULT_CITY) -> dict:
        
        if city not in settings.USE_CITIES:
                return JSONResponse(status_code=404, 
                                    content={"message": f"Указанного города нет в списке: {settings.USE_CITIES}"})
        
        images_names_dict = main_predictor.topk_images_by_text(text=text,
                                                               city=city)

        return images_names_dict