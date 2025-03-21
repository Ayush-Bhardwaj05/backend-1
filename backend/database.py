from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection URL
DATABASE_URL = "mongodb+srv://harshdaftari:harshdaftari123@cluster0.cz6dg.mongodb.net/"
DATABASE_NAME = "my_database"
COLLECTION_NAME = "my_collection"

client = None
db = None
collection = None

async def init_db():
    global client, db, collection
    client = AsyncIOMotorClient(DATABASE_URL)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    print("✅ Database connection established successfully.")

async def close_db():
    if client:
        client.close()
        print("❎ Database connection closed.")
