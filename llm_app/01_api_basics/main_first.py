from fastapi import FastAPI
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

# ===============路径参数学习===============

# 自定义枚举
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
    if model_name.value == "lenet":
        return {"nodel_name": model_name, "message": "LeCNN al the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# 展示定义顺序影响
@app.get("/users/me")
async def read_user_me():
    return {"user_id", "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}

# 路径转换器
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

# ===============查询参数学习===============

fake_iten_db = [{"item_name": "Foo"}, {"iten_name": "Bar"}, {"item_name": "Baz"}]

# 查询参数
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 0):
    return fake_iten_db[skip: skip + limit]

# 可选参数
@app.get("/items/{item_id}")
async def read_item(item_id: str, q : str = None):
    if q:
        return {"item_id": item_id, "q": q}

    return {"item_id": item_id}

# 类型转换
@app.get("/items_r/{item_id}")
async def read_item(item_id: str, q : str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"decription": "This is an amazing item that has a long description"})

    return item

# 多参数查询
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short : bool = False):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"decription": "This is an amazing item that has a long description"})

    return item

# ===============请求体学习===============

# 创建请求体
from pydantic import BaseModel

class Item(BaseModel):
    name : str
    description : str | None = None
    price : float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

# 请求体+路径参数+查询参数
@app.post("/items/{item_id}")
async def update_item(item_id: int, item: Item, q : str | None = None):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result.update({"q": q})
    return result