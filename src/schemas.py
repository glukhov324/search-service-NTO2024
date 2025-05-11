from pydantic import BaseModel
from typing import List



class SightCategoryProbSample(BaseModel):
    category: str
    probability: float

class SightNameCoordsSample(BaseModel):
    sight_name: str
    lon: float
    lat: float

class SearchByImageResult(BaseModel):
    cats_probs: List[SightCategoryProbSample]
    names_coords: List[SightNameCoordsSample]

class SearchByTextSample(SightNameCoordsSample):
    encoded_img: bytes

class SearchByTextResult(BaseModel):
    result: List[SearchByTextSample]