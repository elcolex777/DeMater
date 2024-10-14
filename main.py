from typing import Union

from fastapi import FastAPI
from demater import DeMater

app = FastAPI()


@app.get("/")
def read_root():
    #demater = DeMater(model_path="models\\vosk-model-ru-0.42")
    #demater = DeMater(model_path="models\\vosk-model-small-en-us-0.15")
    demater = DeMater()
    result = demater.process()
    return result


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}