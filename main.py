from fastapi import FastAPI, UploadFile, File
from unsloth import FastVisionModel
import torch
import shutil

app = FastAPI()

model, processor = FastVisionModel.from_pretrained("unsloth/gemma-3n-e2b-it", load_in_4bit=True)
model.generation_config.cache_implementation = "static"

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

    outputs = model.generate(**input_ids, max_new_tokens=16)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"text": result}
