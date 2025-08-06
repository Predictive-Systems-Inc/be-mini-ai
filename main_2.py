# =============================================================================
# FastAPI WebSocket Server for Audio Transcription and AI Teacher Response
# =============================================================================
# This server provides a Gemini-compatible WebSocket interface for real-time
# audio transcription and AI-powered teacher responses for educational conversations.

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from unsloth import FastVisionModel
import torch
import shutil
import os
import json
import base64
import tempfile
import logging
import time
from typing import Dict, Any
import uuid
import asyncio

import numpy as np
from scipy import signal

import librosa

# =============================================================================
# CONSTANTS
# =============================================================================
# Model generation parameters (fallback values if not provided in setup)
TRANSCRIPTION_MAX_TOKENS = 64
TRANSCRIPTION_TEMPERATURE = 0.1
GREETING_MAX_TOKENS = 64
GREETING_TEMPERATURE = 0.7

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TorchInductor cache directory for model optimization
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"

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
# MODEL LOADING
# =============================================================================
# Load the vision-language model at startup for audio transcription and text generation
# This model can handle both audio input (for transcription) and text input (for responses)
model, processor = FastVisionModel.from_pretrained("unsloth/gemma-3n-e2b-it", load_in_4bit=True)
model.generation_config.cache_implementation = "static"

# =============================================================================
# AUDIO CACHE DIRECTORY SETUP
# =============================================================================
# Create a dedicated directory for audio files instead of using system temp
AUDIO_CACHE_DIR = "./audio_cache"
if not os.path.exists(AUDIO_CACHE_DIR):
    os.makedirs(AUDIO_CACHE_DIR)
    logger.info(f"Created audio cache directory: {AUDIO_CACHE_DIR}")

async def check_silence_periodically(client_id: str):
    """
    Background task that periodically checks for silence and processes audio when silence is detected.
    """
    while client_id in manager.connection_states:
        try:
            state = manager.connection_states[client_id]
            
            # Only check if we have accumulated audio and setup is complete
            if state["accumulated_audio"] and state["setup_complete"] and state["last_audio_time"] is not None:
                current_time = time.time() * 1000
                time_since_last_audio = current_time - state["last_audio_time"]
                
                # Calculate total audio size for debugging
                total_base64_size = sum(len(chunk) for chunk in state["accumulated_audio"])
                estimated_audio_size = int(total_base64_size * 0.75)  # Base64 is ~75% of original size
                
                # Check if silence threshold has been reached AND we have enough audio
                min_audio_size = 8000  # Minimum 8KB of audio data
                if (time_since_last_audio >= state["silence_threshold_ms"] and 
                    estimated_audio_size >= min_audio_size):
                    
                    logger.info(f"Silence detected for client {client_id}: {time_since_last_audio:.1f}ms since last audio")
                    logger.info(f"Processing {len(state['accumulated_audio'])} chunks, total base64 size: {total_base64_size}, estimated audio size: {estimated_audio_size}")
                    
                    # Create a copy of the audio chunks to avoid race conditions
                    audio_chunks_to_process = state["accumulated_audio"].copy()
                    
                    # Clear accumulated audio immediately to prevent double processing
                    state["accumulated_audio"] = []
                    state["last_audio_time"] = None
                    
                    # Process the accumulated audio
                    try:
                        # Step 1: Transcribe the accumulated audio
                        transcription = await transcribe_audio_chunks(
                            audio_chunks_to_process,
                            state["temperature"],
                            state["max_output_tokens"]
                        )
                        
                        if transcription.strip() and transcription != "I'm sorry, but I am unable to transcribe the audio as it appears to be gibberish. It doesn't contain any recognizable words or phrases.":
                            logger.info(f"Transcribed: {transcription}")
                            
                            # Step 2: Generate teacher response based on system instruction and context
                            teacher_response = await generate_teacher_response(
                                transcription, 
                                state["system_instruction"],
                                state["conversation_history"],
                                state["temperature"],
                                state["max_output_tokens"]
                            )
                            
                            # Step 3: Update conversation history for context
                            state["conversation_history"].extend([
                                {"role": "user", "content": transcription},
                                {"role": "assistant", "content": teacher_response}
                            ])
                            
                            # Step 4: Send response in Gemini-compatible format
                            await manager.send_message(client_id, {
                                "serverContent": {
                                    "modelTurn": {
                                        "parts": [{"text": teacher_response}]
                                    },
                                    "turnComplete": True
                                }
                            })
                        else:
                            logger.warning("Transcription failed or returned gibberish, skipping response generation")
                            
                    except Exception as e:
                        logger.error(f"Error processing audio chunks: {e}")
                        # If processing failed, restore the audio chunks for retry
                        state["accumulated_audio"] = audio_chunks_to_process
                        state["last_audio_time"] = current_time
                
                elif time_since_last_audio >= state["silence_threshold_ms"]:
                    # Silence detected but not enough audio - clear and reset
                    logger.info(f"Silence detected but insufficient audio ({estimated_audio_size} bytes < {min_audio_size} bytes), clearing chunks")
                    state["accumulated_audio"] = []
                    state["last_audio_time"] = None
            
            # Wait 50ms before checking again
            await asyncio.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error in silence detection for client {client_id}: {e}")
            break

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
            "last_audio_time": None,        # Timestamp of last audio chunk received
            "silence_threshold_ms": 150,    # Silence threshold in milliseconds
            "system_instruction": None,     # Teacher role/system prompt
            "conversation_history": [],      # Chat history for context
            "temperature": TRANSCRIPTION_TEMPERATURE, # Default temperature
            "max_output_tokens": TRANSCRIPTION_MAX_TOKENS # Default max_output_tokens
        }

        # Start background silence detection task
        asyncio.create_task(check_silence_periodically(client_id))
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
    return {"message": "Gemini-Compatible Server Running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "active_connections": len(manager.active_connections)
    }

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    HTTP endpoint for audio transcription (for testing/debugging).
    This endpoint processes uploaded audio files and returns transcription.
    """
    # Save uploaded file to temporary location
    filepath = f"/tmp/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prepare message for the vision-language model
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": filepath},  # Audio input
            {"type": "text", "text": "Transcribe this audio"},  # Instruction
        ] 
    }]

    # Process with the model
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # Generate transcription
    outputs = model.generate(**input_ids, max_new_tokens=TRANSCRIPTION_MAX_TOKENS, do_sample=False, temperature=TRANSCRIPTION_TEMPERATURE)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
    
    # Clean up temporary file
    if os.path.exists(filepath):
        os.remove(filepath)
    
    return {"text": result}

# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/psi.be.v0.gemma3n.GenerateContent")
async def gemini_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint that mimics Gemini's bidirectional streaming interface.
    Handles real-time audio streaming, transcription, and AI teacher responses.
    """
    # Generate unique client ID for this connection
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    try:
        # Main message handling loop
        while True:
            # Receive JSON message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message based on its type
            await handle_message(client_id, message)
            
    except WebSocketDisconnect:
        # Client disconnected normally
        manager.disconnect(client_id)
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

# =============================================================================
# MESSAGE HANDLING
# =============================================================================

async def handle_message(client_id: str, message: dict):
    """
    Route and process different types of WebSocket messages.
    Handles setup, audio streaming, and generates appropriate responses.
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
            
            # Extract generation config parameters from setup
            generation_config = setup_data.get("generation_config", {})
            state["temperature"] = generation_config.get("temperature", state["temperature"])
            state["max_output_tokens"] = generation_config.get("max_output_tokens", state["max_output_tokens"])
            
            logger.info(f"Using temperature: {state['temperature']}, max_output_tokens: {state['max_output_tokens']}")
            
            state["setup_complete"] = True
            
            # Send setupComplete response (client waits for this confirmation)
            await manager.send_message(client_id, {"setupComplete": True})
            
            # Generate and send initial teacher greeting
            greeting = await generate_teacher_greeting(state["system_instruction"])
            
            # Send initial response in Gemini-compatible format
            await manager.send_message(client_id, {
                "serverContent": {
                    "modelTurn": {
                        "parts": [{"text": greeting}]
                    },
                    "turnComplete": True
                }
            })
            return
        
        # =====================================================================
        # AUDIO STREAMING HANDLING
        # =====================================================================
        # Handle real-time audio chunks (sent by client's sendMediaChunk())
        if "realtime_input" in message:
            media_chunks = message["realtime_input"].get("media_chunks", [])
            logger.info(f"Received {len(media_chunks)} media chunks from client {client_id}")
            
            # Process audio chunks with volume detection
            audio_chunks_added = 0
            silent_chunks_ignored = 0
            consecutive_silent_chunks = 0

            for chunk in media_chunks:
                mime_type = chunk.get("mime_type", "")
                data = chunk.get("data", "")
                
                # Only process audio/pcm chunks, skip everything else
                if mime_type == "audio/pcm" and data:
                    try:
                        # Decode the base64 audio data
                        audio_bytes = base64.b64decode(data)
                        
                        # Analyze volume of this chunk using librosa
                        volume, is_silent = analyze_audio_volume(audio_bytes)
                        
                        if is_silent:
                            # Count consecutive silent chunks
                            consecutive_silent_chunks += 1
                            silent_chunks_ignored += 1
                            logger.debug(f"Ignored silent chunk, volume: {volume:.4f}, consecutive silent: {consecutive_silent_chunks}")
                        
                            
                            # If we have accumulated audio and detect a period of silence, process it
                            if (state["accumulated_audio"] and 
                                state["setup_complete"] and 
                                consecutive_silent_chunks >= 3):  # 3 consecutive silent chunks = end of speech
                                
                                logger.info(f"Detected end of speech after {consecutive_silent_chunks} consecutive silent chunks")
                                
                                try:
                                    # Step 1: Transcribe the accumulated audio
                                    transcription = await transcribe_audio_chunks(
                                        state["accumulated_audio"],
                                        state["temperature"],
                                        state["max_output_tokens"]
                                    )
                                    
                                    if transcription.strip() and transcription != "I'm sorry, but I am unable to transcribe the audio as it appears to be gibberish. It doesn't contain any recognizable words or phrases.":
                                        logger.info(f"Transcribed: {transcription}")
                                        
                                        # Step 2: Generate teacher response based on system instruction and context
                                        teacher_response = await generate_teacher_response(
                                            transcription, 
                                            state["system_instruction"],
                                            state["conversation_history"],
                                            state["temperature"],
                                            state["max_output_tokens"]
                                        )
                                        
                                        # Step 3: Update conversation history for context
                                        state["conversation_history"].extend([
                                            {"role": "user", "content": transcription},
                                            {"role": "assistant", "content": teacher_response}
                                        ])
                                        
                                        # Step 4: Send response in Gemini-compatible format
                                        await manager.send_message(client_id, {
                                            "serverContent": {
                                                "modelTurn": {
                                                    "parts": [{"text": teacher_response}]
                                                },
                                                "turnComplete": True
                                            }
                                        })
                                    else:
                                        logger.warning("Transcription failed or returned gibberish, skipping response generation")
                                        
                                except Exception as e:
                                    logger.error(f"Error processing audio chunks: {e}")
                                
                                # Clear accumulated audio for next turn
                                state["accumulated_audio"] = []
                                consecutive_silent_chunks = 0
                                
                        else:
                            # Reset consecutive silent count when we get non-silent audio
                            consecutive_silent_chunks = 0
                            
                            # Add non-silent chunks to accumulation
                            state["accumulated_audio"].append(data)
                            audio_chunks_added += 1
                            logger.debug(f"Added audio chunk, volume: {volume:.4f}, data length: {len(data)}")
                            
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        # Still add the chunk if we can't analyze it
                        state["accumulated_audio"].append(data)
                        audio_chunks_added += 1
                else:
                    logger.warning(f"Skipping non-audio chunk with mime_type: {mime_type}")
            
            if silent_chunks_ignored > 0:
                logger.info(f"Added {audio_chunks_added} audio chunks, ignored {silent_chunks_ignored} silent chunks, consecutive silent: {consecutive_silent_chunks}")
            else:
                logger.info(f"Added {audio_chunks_added} audio chunks")
            
            return
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await manager.send_message(client_id, {
            "error": {"message": str(e)}
        })

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def analyze_audio_volume(audio_data: bytes) -> tuple[float, bool]:
    """
    Analyze audio data using librosa for professional-grade volume detection.
    
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
        
        # Use librosa's built-in silence detection
        # This is much more sophisticated than our custom implementation
        silence_threshold_db = -40  # -40dB threshold (very quiet)
        
        # Calculate RMS energy in dB
        rms_db = librosa.feature.rms(y=audio_normalized, frame_length=2048, hop_length=512)
        rms_db = 20 * np.log10(rms_db + 1e-10)  # Convert to dB
        
        # Check if any frame is above the silence threshold
        has_audio = np.any(rms_db > silence_threshold_db)
        
        # Calculate average volume level
        avg_volume = np.mean(rms_db)
        normalized_volume = (avg_volume + 60) / 60  # Convert from dB to 0-1 scale
        
        # Alternative: Use librosa's voice activity detection
        # This is even more sophisticated for speech detection
        try:
            # Detect voice activity using spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_normalized, sr=16000)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_normalized, sr=16000)
            
            # Voice activity based on spectral characteristics
            voice_detected = np.any(spectral_centroid > 1000) and np.any(spectral_rolloff > 4000)
            
            is_silent = not voice_detected
        except:
            # Fallback to RMS-based detection
            is_silent = not has_audio
        
        logger.debug(f"Librosa analysis - Avg dB: {avg_volume:.2f}, Volume: {normalized_volume:.4f}, Silent: {is_silent}")
        
        return normalized_volume, is_silent
        
    except Exception as e:
        logger.error(f"Error analyzing audio volume with librosa: {e}")
        return 0.0, True

def combine_audio_chunks_with_volume(audio_chunks: list) -> tuple[bytes, float]:
    """
    Combine audio chunks and calculate overall volume.
    
    Args:
        audio_chunks: List of base64-encoded audio data chunks
        
    Returns:
        tuple: (combined_audio_bytes, average_volume)
    """
    if not audio_chunks:
        return b"", 0.0
    
    try:
        # Combine all audio chunks
        combined_base64 = "".join([str(chunk) for chunk in audio_chunks])
        combined_audio = base64.b64decode(combined_base64)
        
        # Analyze volume of combined audio
        volume, is_silent = analyze_audio_volume(combined_audio)
        
        logger.info(f"Combined {len(audio_chunks)} chunks, total size: {len(combined_audio)} bytes, volume: {volume:.4f}, silent: {is_silent}")
        
        return combined_audio, volume
        
    except Exception as e:
        logger.error(f"Error combining audio chunks: {e}")
        return b"", 0.0

async def transcribe_audio_chunks(audio_chunks: list, temperature: float, max_output_tokens: int) -> str:
    """
    Transcribe accumulated audio chunks using the gemma3n model.
    
    Args:
        audio_chunks: List of base64-encoded audio data chunks
        temperature: Generation temperature
        max_output_tokens: Maximum number of tokens for generation
        
    Returns:
        str: Transcribed text from the audio
    """
    if not audio_chunks:
        return ""
    
    try:
        logger.info(f"Receiving audio chunks {len(audio_chunks)} long.")
        
        # Combine audio chunks and analyze volume
        combined_audio, volume = combine_audio_chunks_with_volume(audio_chunks)
        
        if len(combined_audio) == 0:
            logger.warning("No valid audio data after combining chunks")
            return ""
        
        # Check if the combined audio is still silent
        _, is_silent = analyze_audio_volume(combined_audio)
        if is_silent:
            logger.warning("Combined audio is silent, skipping transcription")
            return ""
        
        logger.info(f"Combined audio data size: {len(combined_audio)} bytes, volume: {volume:.4f}")
        
        # Validate audio data
        if len(combined_audio) < 100:  # Too small to be valid audio
            logger.warning(f"Audio data too small: {len(combined_audio)} bytes")
            return ""
        
        # Create WAV file in controlled directory instead of temp folder
        import wave
        import struct
        
        # Generate unique filename
        import uuid
        filename = f"audio_{uuid.uuid4().hex[:8]}_vol{volume:.3f}.wav"
        filepath = os.path.join(AUDIO_CACHE_DIR, filename)
        
        # Create WAV file header
        sample_rate = 16000  # 16kHz as used by the client
        num_channels = 1     # Mono
        sample_width = 2     # 16-bit
        
        # Calculate WAV header
        num_frames = len(combined_audio) // 2  # 16-bit = 2 bytes per sample
        data_size = num_frames * num_channels * sample_width
        file_size = 36 + data_size
        
        logger.info(f"Audio details: {num_frames} frames, {sample_rate}Hz, {num_channels} channels, {sample_width*8} bits")
        
        # Write WAV file directly to controlled directory
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
            wav_file.write(combined_audio)
        
        logger.info(f"Created WAV file: {filepath} ({len(combined_audio)} bytes PCM -> WAV)")
        
        try:
            # Use the same transcription logic as the HTTP endpoint
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": filepath},
                    {"type": "text", "text": "Transcribe this audio"},
                ] 
            }]

            # Process with the model
            input_ids = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=model.dtype)

            # Generate transcription
            outputs = model.generate(**input_ids, max_new_tokens=max_output_tokens, do_sample=False, temperature=temperature)
            result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            result = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
            
            logger.info(f"Raw transcription result: '{result}'")
            return result
            
        finally:
            # Clean up WAV file
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            #     logger.info(f"Cleaned up WAV file: {filepath}")
            # Keep WAV file for investigation - no cleanup
            logger.info(f"WAV file preserved for investigation: {filepath}")    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""

# =============================================================================
# AI RESPONSE GENERATION
# =============================================================================

async def generate_teacher_response(student_input: str, system_parts: list, history: list, temperature: float, max_output_tokens: int) -> str:
    """
    Generate an appropriate teacher response based on student input and context.
    
    Args:
        student_input: The transcribed student speech
        system_parts: System instruction defining teacher role
        history: Previous conversation context
        temperature: Generation temperature
        max_output_tokens: Maximum number of tokens for generation
        
    Returns:
        str: Generated teacher response
    """
    try:
        # Convert system instruction parts to a coherent prompt
        system_prompt = " ".join([part.get("text", "") for part in system_parts])
        
        # Build conversation context with appropriate history
        if not history:
            # First interaction - simple prompt
            prompt_text = f"{system_prompt}\n\nStudent said: {student_input}\n\nRespond as the teacher:"
        else:
            # Include recent conversation history for context
            context = ""
            for msg in history[-4:]:  # Keep last 2 exchanges (4 messages)
                context += f"{msg['role']}: {msg['content']}\n"
            prompt_text = f"{system_prompt}\n\nConversation:\n{context}\nStudent: {student_input}\n\nTeacher:"
        
        # Prepare message for the model
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }]
        
        # Process with the model
        input_ids = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        
        # Generate response with appropriate parameters
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_output_tokens,  # Use max_output_tokens from setup
            do_sample=True,
            temperature=temperature,    # Use temperature from setup
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        # Extract and clean the generated response
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        response = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Teacher response generation failed: {e}")
        return "I'm sorry, I didn't understand that. Could you please try again?"

async def generate_teacher_greeting(system_parts: list) -> str:
    """
    Generate an initial teacher greeting based on the system instruction.
    
    Args:
        system_parts: System instruction defining teacher role
        
    Returns:
        str: Generated greeting message
    """
    try:
        # Convert system instruction to greeting prompt
        system_prompt = " ".join([part.get("text", "") for part in system_parts])
        prompt_text = f"{system_prompt}\n\nGreet the student and in preparation for subject review."
        
        # Prepare message for the model
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }]
        
        # Process with the model
        input_ids = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        
        # Generate greeting with more creative parameters
        outputs = model.generate(
            **input_ids,
            max_new_tokens=GREETING_MAX_TOKENS,
            do_sample=True,
            temperature=GREETING_TEMPERATURE  # Higher temperature for more creative greeting
        )
        
        # Extract and clean the generated greeting
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        greeting = result.split("model\n")[-1].split("<end_of_turn>")[0].strip()

        logger.info(f"Sending TeacherGreeting: {greeting}")
        
        return greeting
        
    except Exception as e:
        logger.error(f"Greeting generation failed: {e}")
        return "Hello! Let's start our physics lesson. What would you like to learn about today?"

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    # Start the server on all interfaces (0.0.0.0) on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)