from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from unsloth import FastVisionModel
import torch
import shutil
import os
import json
import base64
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"

app = FastAPI()

# Add CORS for WebSocket
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup (same as your original)
model, processor = FastVisionModel.from_pretrained("unsloth/gemma-3n-e2b-it", load_in_4bit=True)
model.generation_config.cache_implementation = "static"

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": str(model.device) if model else "none"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    filepath = f"/tmp/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": filepath},
            {"type": "text", "text": "Transcribe this audio"},
        ] 
    }]

    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    outputs = model.generate(**input_ids, max_new_tokens=64, do_sample=False, temperature=0.1)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
    
    return {"text": result}

# Simple WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received message: {message}")
            
            # Handle audio data
            if "audio_data" in message:
                audio_b64 = message["audio_data"]
                mime_type = message.get("mime_type", "audio/wav")
                
                try:
                    # Use your exact transcribe logic
                    transcription = await transcribe_base64_audio(audio_b64, mime_type)
                    
                    # Send response
                    response = {
                        "type": "transcription",
                        "text": transcription
                    }
                    await websocket.send_text(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
            
            # Handle ping/pong
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Unknown message format"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def transcribe_base64_audio(audio_b64: str, mime_type: str) -> str:
    """Use your exact transcribe logic but with base64 audio data"""
    
    # Convert base64 to file (same as your transcribe logic)
    audio_data = base64.b64decode(audio_b64)
    
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_data)
        filepath = temp_file.name
    
    try:
        # Your exact transcribe logic
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": filepath},
                {"type": "text", "text": "Transcribe this audio"},
            ] 
        }]

        input_ids = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**input_ids, max_new_tokens=64, do_sample=False, temperature=0.1)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(result)
        result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
        
        return result
        
    finally:
        # Cleanup temp file
        if os.path.exists(filepath):
            os.remove(filepath)