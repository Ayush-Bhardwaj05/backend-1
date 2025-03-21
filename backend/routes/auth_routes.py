from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorCollection
from utils import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,hash_password
from datetime import timedelta

router = APIRouter()

# Pydantic Model
class User(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Pass collection from main.py
collection: AsyncIOMotorCollection = None

def initialize_collection(col):
    global collection
    collection = col

@router.post("/auth/signup")
async def signup(user: User):
    if collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized.")
    
    existing_user = await collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists.")
    
    user.password = hash_password(user.password)
    await collection.insert_one(user.model_dump())
    return {"message": "User registered successfully"}

@router.post("/auth/login", response_model=Token)
async def login(user: User):
    if collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized.")
    
    db_user = await collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user['password']):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(
        data={"sub": db_user['email']},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}
