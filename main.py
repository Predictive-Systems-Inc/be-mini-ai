from fastapi import FastAPI, UploadFile, File
from unsloth import FastVisionModel
import torch
import shutil
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"

app = FastAPI()

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

    # Generate output from the model
    outputs = model.generate(**input_ids, max_new_tokens=64, do_sample=False,
        temperature=0.1)

    # decode and print the output as text
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only transcription
    result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
    return {"text": result}


