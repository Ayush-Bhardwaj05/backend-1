from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import aiofiles
import tempfile
import traceback
from pydub import AudioSegment
import uuid

# Create a FastAPI router
convo_router = APIRouter(prefix="/convo", tags=["Conversation"])

# Configure the API key
genai.configure(api_key="AIzaSyCyVeQ2RU1jmNjbLmgTSuivNRFEz-aSjio")

UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TextRequest(BaseModel):
    text: str

@convo_router.post("/process-text")
async def process_text(data: TextRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        content_response = model.generate_content(data.text)
        return {"response": content_response.text}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❗ Error in process-text: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_webm_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path, format="webm")
        audio.export(output_path, format="wav")
        print(f"✅ Converted to WAV: {output_path}")
        return output_path
    except Exception as e:
        print(f"❗ Error during conversion: {e}")
        raise HTTPException(status_code=500, detail="Error converting file to WAV.")

@convo_router.post("/process-audio")
async def upload_audio(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    print(f"File saved at: {file_path}")  # Debug print

    # Process the audio file with genai
    try:
        myfile = genai.upload_file(file_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            contents=['Describe this emotion like angry, sad, or happy', myfile]
        )
        print(f"Gemini response: {response.text}", flush=True)  # Print the response in the terminal
        return {"message": "File uploaded and processed", "file_path": filename, "response": response.text}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❗ Error in process-audio: {error_details}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@convo_router.get("/get-audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}