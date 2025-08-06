# =============================================================================
# FastAPI WebSocket Server for Audio Testing (Simplified Version)
# =============================================================================
# This server provides a simplified WebSocket interface for testing audio file saving
# without the unsloth model dependencies.

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from unsloth import FastVisionModel
from transformers import AutoProcessor, AutoModelForImageTextToText

import os
import json
import base64
import logging
import time
from typing import Dict, Any
import uuid
import asyncio
from json_reader import get_question, build_response_prompt, find_node_by_tag
import numpy as np
import librosa
import json
import random

GEMMA_MAX_TOKENS = 50
GEMMA_TEMPERATURE = 0.1
#GEMMA_MODEL_ID = "unsloth/gemma-3n-e2b-it"
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

TEACHER_NAME = "teacher_1"

# =============================================================================
# CONSTANTS
# =============================================================================
# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI()

# Configure CORS to allow cross-origin requests (needed for WebSocket clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# AUDIO CACHE DIRECTORY SETUP
# =============================================================================
# Create a dedicated directory for audio files instead of using system temp
AUDIO_CACHE_DIR = "./audio_cache"
if not os.path.exists(AUDIO_CACHE_DIR):
    os.makedirs(AUDIO_CACHE_DIR)
    logger.info(f"Created audio cache directory: {AUDIO_CACHE_DIR}")


# =============================================================================
# MODEL LOADING
# =============================================================================
# Load the vision-language model at startup for audio transcription and text generation
# This model can handle both audio input (for transcription) and text input (for responses)

# model, processor = FastVisionModel.from_pretrained("unsloth/gemma-3n-e2b-it", load_in_4bit=True)
# model.generation_config.cache_implementation = "static"

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="auto")
model = AutoModelForImageTextToText.from_pretrained(
            GEMMA_MODEL_ID, torch_dtype="auto", device_map="auto")


def analyze_audio_volume(audio_data: bytes) -> tuple[float, bool]:
    """
    Analyze audio data using librosa for volume-based silence detection.
    
    Args:
        audio_data: Raw PCM audio data (16-bit, 16kHz, mono)
        
    Returns:
        tuple: (volume_level, is_silent)
    """
    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1] range (librosa expects this)
        audio_normalized = audio_array / 32768.0
        
        # Calculate RMS energy in dB
        rms_db = librosa.feature.rms(y=audio_normalized, frame_length=2048, hop_length=512)
        rms_db = 20 * np.log10(rms_db + 1e-10)  # Convert to dB
        
        # Calculate average volume level in dB
        avg_volume_db = float(rms_db.mean())
        
        # Convert dB to a 0-1 scale (typical range is -60dB to 0dB)
        normalized_volume = (avg_volume_db + 60) / 60
        normalized_volume = max(0.0, min(1.0, normalized_volume))
        
        # Simple volume-based silence detection
        silence_threshold_db = 0.2
        is_silent = normalized_volume < silence_threshold_db
        
        return normalized_volume, is_silent
        
    except Exception as e:
        logger.error(f"Error analyzing audio volume with librosa: {e}")
        return 0.0, True

# =============================================================================
# WEBSOCKET CONNECTION MANAGEMENT
# =============================================================================
class ConnectionManager:
    """
    Manages WebSocket connections and their associated state.
    Handles multiple concurrent clients with individual conversation states.
    """
    def __init__(self):
        # Store active WebSocket connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Store conversation state for each client
        self.connection_states: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection and initialize client state"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Initialize conversation state for this client
        self.connection_states[client_id] = {
            "setup_complete": False,        # Whether initial setup is done
            "accumulated_audio": [],        # Audio chunks waiting to be processed
            "system_instruction": None,     # Teacher role/system prompt
            "conversation_history": [],      # Chat history for context,
            "silence_start_time": None,     # Track when silence began
            "question_index": 0,            # Current question index
            "current_node": None,           # Current conversation node
        }

        # Remove global variable initialization since they're now in state
        # global question_index
        # global current_node
        # question_index = 0
        # current_node = None
        
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Clean up resources when a client disconnects"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_states:
            del self.connection_states[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        """Send a JSON message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

# Create global connection manager instance
manager = ConnectionManager()

# =============================================================================
# HTTP ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint to verify server is running"""
    return {"message": "Audio Testing Server Running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections)
    }

@app.websocket("/ws/test")
async def test_websocket(websocket: WebSocket):
    """Simple test WebSocket endpoint"""
    await websocket.accept()
    logger.info("Test WebSocket connection established")
    
    try:
        await websocket.send_text("Test WebSocket is working!")
        await websocket.close()
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")

# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/psi.be.v0.gemma3n.GenerateContent")
async def gemini_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint that mimics Gemini's bidirectional streaming interface.
    Handles real-time audio streaming and file saving for testing.
    """
    logger.info("WebSocket connection attempt received")
    
    # Generate unique client ID for this connection
    client_id = str(uuid.uuid4())
    logger.info(f"Generated client ID: {client_id}")
    
    try:
        await manager.connect(websocket, client_id)
        logger.info(f"WebSocket connection established for client {client_id}")
        
        # Main message handling loop
        while True:
            # Receive JSON message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message based on its type
            await handle_message(client_id, message)
            
    except WebSocketDisconnect:
        # Client disconnected normally
        logger.info(f"Client {client_id} disconnected normally")
        manager.disconnect(client_id)
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

# =============================================================================
# MESSAGE HANDLING
# =============================================================================

# async def generate_response(client_id: str, prompt: str, filepath: str):
#     """
#     Generate a response using the gemma model.
#     """
#     try:
#         # Use the same transcription logic as the HTTP endpoint
#         messages = [
#             {
#             "role": "user",
#             "content": [
#                 {"type": "audio", "audio": filepath},
#                 {"type": "text", "text": prompt},
#             ] 
#         }]

#         # Process with the model
#         input_ids = processor.apply_chat_template(
#             messages, add_generation_prompt=True,
#             tokenize=True, return_dict=True, return_tensors="pt"
#         ).to(model.device, dtype=model.dtype)

#         # Generate transcription
#         outputs = model.generate(**input_ids, 
#                                     max_new_tokens=GEMMA_MAX_TOKENS,
#                                     do_sample=False, 
#                                     temperature=GEMMA_TEMPERATURE)
#         result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#         result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
        
#         logger.info(f"Raw transcription result: '{result}'")
#         return result
        
#     finally:
#         # Clean up WAV file
#         if os.path.exists(filepath):
#             os.remove(filepath)

async def generate_response(client_id: str, prompt: str, filepath: str):
    """
    Stream a response using the Gemma3n model with audio input.
    """
    try:
        # Load and process audio
        audio_inputs = processor(audio=filepath, return_tensors="pt")
        audio_features = model.get_audio_embeddings(audio_inputs.input_values)

        # Get placeholder mask for audio
        placeholder_mask = model.get_placeholder_mask(audio_features=audio_features)

        # Prepare chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": filepath},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        # Tokenize input
        input_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=model.dtype)

        # Stream generate
        stream = model.generate_stream(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            audio_features=audio_features,
            placeholder_mask=placeholder_mask,
            max_new_tokens=GEMMA_MAX_TOKENS,
            do_sample=False,
            temperature=GEMMA_TEMPERATURE
        )

        result = ""
        async for token in stream:
            result += processor.decode(token, skip_special_tokens=True)

        result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
        logger.info(f"Raw transcription result: '{result}'")
        return result

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

        
async def handle_message(client_id: str, message: dict):
    """
    Route and process different types of WebSocket messages.
    Handles setup and audio streaming.
    """
    state = manager.connection_states.get(client_id, {})
    
    try:
        # =====================================================================
        # SETUP MESSAGE HANDLING
        # =====================================================================
        # Handle initial setup message (sent by client's sendInitialSetup())
        if "setup" in message:
            logger.info(f"Setup received from {client_id}")
            
            # Extract system instruction that defines the teacher's role
            setup_data = message["setup"]
            system_instruction = setup_data.get("system_instruction", {})
            state["system_instruction"] = system_instruction.get("parts", [])
            
            logger.info(f"Setup completed for {client_id}")
            
            state["setup_complete"] = True
            
            # Send setupComplete response (client waits for this confirmation)
            await manager.send_message(client_id, {"setupComplete": True})
            
            # Send initial greeting
            await manager.send_message(client_id, await generate_teacher_greeting(client_id, state))
            return
        
        # =====================================================================
        # AUDIO STREAMING HANDLING
        # =====================================================================
        # Handle real-time audio chunks (sent by client's sendMediaChunk())
        if "realtime_input" in message:
            media_chunks = message["realtime_input"].get("media_chunks", [])

            if not state["setup_complete"]:
                return
            
            # Process audio chunks with volume detection
            for chunk in media_chunks:
                mime_type = chunk.get("mime_type", "")
                data = chunk.get("data", "")
                
                # Only process audio/pcm chunks, skip everything else
                if mime_type == "audio/pcm" and data:
                    # Decode the base64 audio data
                    audio_bytes = base64.b64decode(data)
                    
                    # Analyze volume of this chunk using librosa
                    volume, is_silent = analyze_audio_volume(audio_bytes)
                    
                    if is_silent:
                        # Track silence duration in milliseconds
                        if "silence_start_time" not in state or state["silence_start_time"] is None:
                            state["silence_start_time"] = time.time() * 1000
                        
                        # Calculate how long we've been silent
                        current_time = time.time() * 1000
                        silence_duration_ms = current_time - state["silence_start_time"]
                        
                        # Only save if we have accumulated audio AND there was actual speech
                        if (state["accumulated_audio"] and 
                            state["setup_complete"] and 
                            state.get("has_speech", False) and  # Only save if we had speech
                            silence_duration_ms >= 200):  # 200ms silence threshold
                            
                            logger.info(f"End of speech detected after {silence_duration_ms:.1f}ms of silence")
                            
                            # Save the accumulated audio to file
                            filepath = await save_audio_chunks(state["accumulated_audio"])
                                                        # Clear accumulated audio for next turn
                            state["accumulated_audio"] = []
                            state["silence_start_time"] = None
                            state["has_speech"] = False  # Reset speech flag

                            
                            if filepath:
                                logger.info(f"Audio saved successfully: {filepath}")
                                # Let's build the prompt for the gemma inference
                                prompt, response_messages, tags = build_response_prompt(TEACHER_NAME, state)
                                if not tags:
                                    # no more responses, move to the next question
                                    print("no more responses, move to the next question")
                                    state["question_index"] += 1
                                    state["current_node"] = None
                                    print("state['question_index']", state["question_index"])
                                    # build the prompt for the next question
                                    await manager.send_message(client_id, await generate_teacher_greeting(client_id, state))
                                    return
 
                                # Gemma Inference goes here...
                                response = await generate_response(client_id, prompt, filepath)
                                print("response", response)
                                # find the index of the response in response_messages
                                index = response_messages.index(response)
                                print("index of response", index)
                                tag = tags[index]
                                node = find_node_by_tag(tag)
                                
                                # # Read and send the existing voice file to client
                                # # pick random from responses
                                # print("before random selection.")
                                # print("state['current_node']", state["current_node"])
                                # print("responses", responses)
                                # random_tag = random.choice(tags)
                                # node = find_node_by_tag(random_tag)
                                state["current_node"] = node  # Update state instead of global
                                voice_response = await send_voice_file_to_client(client_id, node["audio link"])
                                
                                if voice_response:
                                    # Send the voice file response
                                    await manager.send_message(client_id, voice_response)
                            else:
                                logger.warning("Failed to save audio file")
                            
                        else:
                            # Keep adding to the accumulated audio, including silence
                            state["accumulated_audio"].append(data)
                            
                    else:
                        # Reset silence tracking when we get non-silent audio
                        state["silence_start_time"] = None
                        state["has_speech"] = True  # Mark that we have speech
                        
                        # Add non-silent chunks to accumulation
                        state["accumulated_audio"].append(data)
                        

                        
                else:
                    logger.warning(f"Skipping non-audio chunk with mime_type: {mime_type}")
                        
            return
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await manager.send_message(client_id, {
            "error": {"message": str(e)}
        })

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

async def save_audio_chunks(audio_chunks: list) -> str:
    """
    Save accumulated audio chunks as a WAV file for debugging.
    """
    if not audio_chunks:
        return ""
    
    try:
        
        # Decode each chunk separately to binary, then combine
        binary_chunks = []
        total_base64_size = 0
        
        for i, chunk in enumerate(audio_chunks):
            # Clean the base64 data for this chunk
            cleaned_chunk = chunk.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            
            # Ensure proper padding for this chunk
            padding_needed = len(cleaned_chunk) % 4
            if padding_needed:
                cleaned_chunk += '=' * (4 - padding_needed)
            
            try:
                # Decode this chunk to binary
                binary_data = base64.b64decode(cleaned_chunk)
                binary_chunks.append(binary_data)
                total_base64_size += len(chunk)
                
            except Exception as e:
                logger.error(f"Failed to decode chunk {i}: {e}")
                continue
        
        # Combine all binary chunks
        audio_data = b''.join(binary_chunks)
                
        # Validate audio data
        if len(audio_data) < 100:  # Too small to be valid audio
            logger.warning(f"Audio data too small: {len(audio_data)} bytes")
            return ""
        
        # Create WAV file
        import wave
        import struct
        
        # Generate unique filename with timestamp
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_count = len(audio_chunks)
        audio_size = len(audio_data)
        filename = f"audio_{timestamp}_chunks{chunk_count}_size{audio_size}_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(AUDIO_CACHE_DIR, filename)
        
        # Create WAV file header
        sample_rate = 16000  # 16kHz as used by the client
        num_channels = 1     # Mono
        sample_width = 2     # 16-bit
        
        # Calculate WAV header
        num_frames = len(audio_data) // 2  # 16-bit = 2 bytes per sample
        data_size = num_frames * num_channels * sample_width
        file_size = 36 + data_size
                
        # Write WAV file
        with open(filepath, 'wb') as wav_file:
            # Write WAV header
            wav_file.write(b'RIFF')
            wav_file.write(struct.pack('<I', file_size))
            wav_file.write(b'WAVE')
            wav_file.write(b'fmt ')
            wav_file.write(struct.pack('<I', 16))  # fmt chunk size
            wav_file.write(struct.pack('<H', 1))   # PCM format
            wav_file.write(struct.pack('<H', num_channels))
            wav_file.write(struct.pack('<I', sample_rate))
            wav_file.write(struct.pack('<I', sample_rate * num_channels * sample_width))  # byte rate
            wav_file.write(struct.pack('<H', num_channels * sample_width))  # block align
            wav_file.write(struct.pack('<H', sample_width * 8))  # bits per sample
            wav_file.write(b'data')
            wav_file.write(struct.pack('<I', data_size))
            
            # Write PCM data
            wav_file.write(audio_data)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Audio saving failed: {e}")
        return ""

async def generate_teacher_greeting(client_id: str, state: dict) -> dict:
    """
    Generate a teacher greeting using the current state.
    """
    # Check if we've run out of questions
    question = get_question(TEACHER_NAME, state["question_index"])
    if question is None:
        # No more questions available
        return {
            "type": "text",
            "content": "Great job! We've completed all the questions. Thank you for participating!"
        }
    
    state["current_node"] = question
    print("question", question)
    return await send_voice_file_to_client(client_id, question["audio"])

async def send_voice_file_to_client(client_id: str, voice_filepath: str) -> dict:
    """
    Read the voice file and send it as PCM data to the client.
    """
    try:
        # Check if the file exists
        if not os.path.exists(voice_filepath):
            logger.warning(f"Voice file not found: {voice_filepath}")
            return None
        
        # Read the WAV file using scipy for better format support
        import scipy.io.wavfile as wavfile
        
        try:
            sample_rate, audio_data = wavfile.read(voice_filepath)
            logger.info(f"Read WAV file: {sample_rate}Hz, shape: {audio_data.shape}")
            
            # Convert to 16-bit PCM if needed
            if audio_data.dtype != np.int16:
                # Normalize and convert to 16-bit
                audio_data = (audio_data.astype(np.float32) / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
                logger.info("Converted audio to 16-bit PCM")
            
            # Resample to 24kHz if needed
            target_sample_rate = 24000
            if sample_rate != target_sample_rate:
                logger.info(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
                from scipy import signal
                
                # Calculate resampling ratio
                ratio = target_sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                
                # Resample using scipy
                resampled_audio = signal.resample(audio_data, new_length)
                
                # Convert back to 16-bit PCM
                resampled_audio = (resampled_audio.astype(np.float32) / np.max(np.abs(resampled_audio)) * 32767).astype(np.int16)
                
                audio_data = resampled_audio
                sample_rate = target_sample_rate
                logger.info(f"Resampled to {sample_rate}Hz")
            
            # Convert to bytes
            pcm_bytes = audio_data.tobytes()
            
        except Exception as e:
            logger.error(f"Error reading with scipy: {e}")
            # Fallback to wave module
            import wave
            
            with wave.open(voice_filepath, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                pcm_bytes = frames
                logger.info(f"Read WAV file (fallback): {sample_rate}Hz")
        
        # Convert PCM data to base64
        pcm_base64 = base64.b64encode(pcm_bytes).decode('utf-8')
        
        logger.info(f"Converted to PCM base64: {len(pcm_base64)} chars, sample rate: {sample_rate}Hz")
        
        # Create response with PCM data
        response = {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": pcm_base64
                            }
                        }
                    ]
                },
                "turnComplete": True
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error reading voice file: {e}")
        return None

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    # Start the server on all interfaces (0.0.0.0) on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080) 