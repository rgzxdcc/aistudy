from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()

# ===============响应模型===============

class Item(BaseModel):
    name : str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []

# @app.post("/items/")
# async def create_item(item: Item)->Item:
#     return item

# @app.get("/items/")
# async def read_items()->list[Item]:
#     return[
#         Item(name= "Portal Gun", price=24.0),
#         Item(name = "Plumbus", price = 32.0),
#     ]

@app.post("/items/", response_model=Item)
async def create_item(item: Item)->Any:
    return item

@app.get("/items/", response_model=list[Item])
async def read_items()->Any:
    return[
        Item(name= "Portal Gun", price=24.0),
        Item(name = "Plumbus", price = 32.0),
    ]


# 参数过滤
class BaseUser(BaseModel):
    username : str
    email : EmailStr
    full_name : str | None = None

class UserIn(BaseUser):
    password : str

@app.post("/user/")
async def create_user(user : UserIn) -> BaseUser:
    return user

# response子类
from fastapi import Response
from fastapi.responses import RedirectResponse, JSONResponse

@app.get("/portal")
async def get_portal(teleport:bool = False)->Response:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    return JSONResponse(content={"message": "Here's your interdimensional portal."})

# 编码参数
items = {
    "foo": {"name": "Foo", "price":50.2},
    "bar": {"name": "Bar", "description":"lalala", "price":62, "tax":20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}

@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items[item_id]


# ===============错误处理===============
from fastapi import HTTPException

single_items = {"foo":"The foo Wrestler"}

@app.get("/items_s/{item_id}")
async def read_item(item_id: str):
    if item_id not in single_items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": single_items[item_id]}

# 自定义异常处理器
from fastapi import Request
from fastapi.responses import JSONResponse

class UnicornException(Exception):
    def __init__(self, name:str):
        self.name = name
    
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(status_code=418, 
        content={"message":f"Oops! {exc.name} did something .There goes a ranbow..."})

@app.get("/unicorn/{name}")
async def read_unicorn(name: str):
    if name == "dcc":
        raise UnicornException(name)
    return {"unicorn_name": name}

# 覆盖默认异常处理器
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

# 覆盖默认异常处理器(HttpException)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

# 覆盖请求验证异常(RequestValidationError)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(requeser, exc: RequestValidationError):
    message="Validation errors: "
    for error in exc.errors():
        message += f"\nField :{error['loc']}, Error:{error['msg']}"
    return PlainTextResponse(message, status_code=400)

@app.get("/items_e/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}

