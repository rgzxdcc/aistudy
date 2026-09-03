from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name : str
    price: float

@app.post("/items/{item_id}")
async def read_item(item_id : int, item : Item)->Item:
    if item_id == 1:
        item.name = "apple"
        return item
    elif item_id == 2:
        raise HTTPException(status_code=400, detail="sorry, i dont like number 2")

    return {"name": "orange", "price": 32}
