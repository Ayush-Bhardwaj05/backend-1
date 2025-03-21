from fastapi import FastAPI, Depends
from routes.predict import router as predict_router
from routes.convo import convo_router
from routes.auth_routes import router as auth_router, initialize_collection
from auth import verify_token
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI()

app.include_router(predict_router)
app.include_router(convo_router)
app.include_router(auth_router)

DATABASE_URL = "mongodb+srv://harshdaftari:harshdaftari123@cluster0.cz6dg.mongodb.net/"
DATABASE_NAME = "my_database"
COLLECTION_NAME = "my_collection"

# Initialize MongoDB
client = None
db = None
collection = None

async def init_db():
    global client, db, collection
    try:
        client = AsyncIOMotorClient(DATABASE_URL)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        initialize_collection(collection)
        print("✅ Database connection established successfully.")
    except Exception as e:
        print(f"❗ Error connecting to database: {e}")

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()
        print("❎ Database connection closed.")

@app.get("/")
def root():
    return {"message": "Welcome to the Speech Emotion Recognition API!"}

@app.get("/protected")
def protected_route(username: str = Depends(verify_token)):
    return {"message": f"Welcome, {username}! This is a protected route."}
