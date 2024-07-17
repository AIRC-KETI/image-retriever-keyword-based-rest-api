from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import tag_retrieval_structural
import utils
import json

app = FastAPI()
configs = json.load(open("configs.json", "r"))

@app.get("/")
async def read_root():
    return "This is root path from image retrieval api"

@app.get("/retrieve_images_with_text_query/")
def retrieve_images_with_text_query(text_query:str):
    #image_url_list, detected_tags_concat, found_image_tags_list = tag_retrieval_query_keyword_extraction.find_image(text_query, return_found_tags=True, return_image_as_url=True, return_bucket_url=True, num_images=5)
    image_url_list, detected_tags_concat, found_image_tags_list = tag_retrieval_structural.find_image(text_query, configs, return_found_tags=True, return_image_as_url=True, return_bucket_url=True, num_images=5)
    return {"image_url_list": image_url_list, "image_tag_list":found_image_tags_list}

@app.get("/retrieve_image_tags_with_id/")
def retrieve_image_tags_with_id(img_id:int):
    return {"img_id":img_id, "img_tags":utils.tags_given_id(img_id)}
