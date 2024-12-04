import os
import uvicorn  # Required for running FastAPI
from fastapi import (
    FastAPI, 
    WebSocket, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    BackgroundTasks, 
    WebSocketDisconnect,
    Request  # Add this import
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import subprocess
from modules.processors.frame.core import get_frame_processors_modules
from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import get_face_swapper
import modules.globals
import torch
import gc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
# Add CORS middleware to allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize empty/None values
global frame_processors
global source_image

source_image = None
frame_processors = []

# Initialize basic configurations without loading any image
modules.globals.frame_processors = []  # Empty initially
modules.globals.execution_providers = ['CUDAExecutionProvider']
modules.globals.source_path = None  # No default image

# ThreadPoolExecutor for parallel frame processing
executor = ThreadPoolExecutor()

user_source_images = {}
# global frame_processors

# Configure CUDA for PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Home route to show a starting message
@app.get("/")
async def read_index(tool: str = None):
    if tool == "image_video_face_swap":
        return FileResponse('static/image_video_face_swap.html')
    elif tool == "live_face_swap":
        return FileResponse('static/live_face_swap.html')
    return FileResponse('static/index.html')


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    if source_image is None:
        logger.error("No source image found")
        await websocket.close(code=1000, reason="No source image uploaded")
        return

    stream_id = int(time.time())
    rtmp_url = f'rtmp://54.242.212.173:1935/live/stream_{stream_id}'
    logger.info(f"Generated RTMP URL: {rtmp_url}")
    
    ffmpeg_process = None
    frame_count = 0

    try:
        logger.debug(f"Starting FFMPEG process with URL: {rtmp_url}")
        ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', '1280x720',
            '-r', '30',
            '-i', 'pipe:0',
            '-c:v', 'h264_nvenc',
            '-preset', 'llhq',
            '-rc', 'cbr',
            '-zerolatency', 'true',
            '-b:v', '4M',
            '-maxrate', '4M',
            '-bufsize', '8M',
            '-profile:v', 'high',
            '-g', '30',
            '-keyint_min', '30',
            '-f', 'flv',
            rtmp_url
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Log FFMPEG output
        async def log_ffmpeg_output():
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, ffmpeg_process.stderr.readline
                )
                if not line:
                    break
                logger.debug(f"FFMPEG: {line.decode().strip()}")

        asyncio.create_task(log_ffmpeg_output())

        while True:
            try:
                data = await websocket.receive_bytes()
                
                if data.startswith(b"VIDEO:"):
                    frame_count += 1
                    video_data = data[6:]
                    frame = cv2.imdecode(np.frombuffer(video_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None and source_image is not None:
                        logger.debug(f"Processing frame {frame_count}")
                        processed_frame = await asyncio.get_event_loop().run_in_executor(
                            executor, process_frame_in_thread, source_image, frame, frame_processors
                        )
                        
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        await websocket.send_bytes(b"VIDEO:" + buffer.tobytes())
                        
                        try:
                            ffmpeg_process.stdin.write(processed_frame.tobytes())
                            ffmpeg_process.stdin.flush()
                            logger.debug(f"Frame {frame_count} written to FFMPEG")
                        except Exception as e:
                            logger.error(f"Error writing to FFMPEG: {e}")

            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if ffmpeg_process:
            try:
                ffmpeg_process.stdin.close()
                ffmpeg_process.terminate()
                await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)
                logger.info("FFMPEG process terminated")
            except Exception as e:
                logger.error(f"Error closing FFMPEG: {e}")

async def process_frames(frame_buffer, websocket, source_image, frame_processors):
    while True:
        try:
            frame = await frame_buffer.get()
            
            # Process frame with face swapping
            processed_frame = await asyncio.get_event_loop().run_in_executor(
                executor, process_frame_in_thread, source_image, frame, frame_processors
            )
            
            # Encode and send processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            await websocket.send_bytes(b"VIDEO:" + buffer.tobytes())
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Frame processing error: {e}")
            continue

# Add RTMP URL generation endpoint
@app.get("/generate-stream-url")
async def generate_stream_url():
    stream_id = int(time.time())
    server_ip = "54.242.212.173"  # Replace with your server IP
    
    urls = {
        "rtmp_url": f"rtmp://{server_ip}:1935/live/stream_{stream_id}",
        "srt_url": f"srt://{server_ip}:9710?mode=caller&latency=120",
        "instructions": {
            "vlc": [
                "1. Open VLC Media Player",
                "2. Press Ctrl+N or go to Media > Open Network Stream",
                "3. Paste the RTMP or SRT URL",
                "4. Click Play"
            ]
        }
    }
    return urls

@app.post("/upload")
async def upload_image(image: dict):
    image_name = image.get("image")
    print(image_name)
    if not image_name:
        raise HTTPException(status_code=400, detail="Image name must be provided.")
    
    modules.globals.source_path = f"C:/FaceSwap/Deep-Live-Cam-main/static/images/{image_name}"

    print(modules.globals.source_path)

    global source_image
    global frame_processors

    modules.globals.frame_processors.append('face_swapper')
    modules.globals.execution_providers.append('CUDAExecutionProvider')
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)  # Load frame processors
    # Load the source image if not already loaded
    source_image = None
    if source_image is None and modules.globals.source_path:
        source_image =  get_one_face(cv2.imread(modules.globals.source_path))
        print(source_image)
    
    print(source_image)
    return JSONResponse(content={"message": f"Image {image_name} uploaded successfully."})



# Endpoint to receive and store the source image for face swapping
@app.post("/upload_source_image")
async def upload_source_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Create a temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save the file with a unique name
        file_path = f"temp/source_image_{int(time.time())}.jpg"
        with open(file_path, "wb") as f:
            f.write(contents)

        # Update the global source path and image
        modules.globals.source_path = file_path
        
        global source_image
        global frame_processors

        # Initialize frame processors only when image is uploaded
        modules.globals.frame_processors = ['face_swapper']  # Reset and add face_swapper
        
        # Reload frame processors
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        
        # Load the new source image
        source_image = get_one_face(cv2.imread(file_path))
        
        if source_image is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        return {"message": "Source image uploaded and processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Function to process frame in a separate thread
def process_frame_in_thread(source_image, frame, frame_processors):
    temp_frame = cv2.flip(frame.copy(), 1)  # Flip frame horizontally
    for frame_processor in frame_processors:
        temp_frame = frame_processor.process_frame(source_image, temp_frame)
    return temp_frame

# WebSocket endpoint for real-time video streaming
@app.websocket("/video_stream")
async def video_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    global source_image
    global frame_processors

    last_frame_time = time.time()
    fps_limit = 15  # Limit to 15 FPS for smoother streaming

    try:
        while True:
            current_time = time.time()
            if current_time - last_frame_time < 1 / fps_limit:
                continue  # Skip frame to match FPS limit

            last_frame_time = current_time
            # Receive frame data from client
            data = await websocket.receive_bytes()
            print(f"Received data size: {len(data)}")
            np_data = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame. Skipping.")
                continue

            if source_image is None:
                print("Source image not found or unreadable.")
                continue

            # Process frame asynchronously using ThreadPoolExecutor
            processed_frame = await asyncio.get_event_loop().run_in_executor(
               executor, process_frame_in_thread, source_image, frame, frame_processors
            )

            print("Done")
            # Compress and send processed frame back to client
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Compress frame
            await websocket.send_bytes(buffer.tobytes())

            del processed_frame

    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        print("WebSocket connection closed")
        await websocket.close()


@app.websocket("/audio_stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Audio WebSocket connection established")

    try:
        while True:
            # Receive audio data from the client
            audio_data = await websocket.receive_bytes()

            if audio_data is None:
                print("Failed to get audio. Skipping.")
                continue

            print(f"Received audio data size: {len(audio_data)}")

            processed_audio = audio_data
            
            # Send back the processed audio data
            await websocket.send_bytes(processed_audio)

    except Exception as e:
        print(f"Error processing audio: {e}")
    finally:
        print("Audio WebSocket connection closed")
        await websocket.close()


@app.post("/process")
async def process_image(
    source: UploadFile, 
    target: UploadFile, 
    output: str = Form(...),
    keep_fps: bool = Form(False),
    keep_audio: bool = Form(True),
    keep_frames: bool = Form(False),
    map_faces: bool = Form(False),
    many_faces: bool = Form(False),
    frame_processor: list[str] = Form(['face_swapper']),
    background_tasks: BackgroundTasks = None
):
    source_path = f"./{source.filename}"
    target_path = f"./{target.filename}"

    # Save the uploaded files to disk
    try:
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(source.file, buffer)

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target.file, buffer)

        # Verify if files are saved correctly
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            raise HTTPException(status_code=400, detail="File not saved correctly")

        # Prepare the command for processing
        command = [
            "python", "./run.py",
            "-s", source_path,
            "-t", target_path,
            "-o", output,
            "--frame-processor", *frame_processor,
            "--execution-provider","cuda"
        ]

        if keep_fps:
            command.append('--keep-fps')
        if keep_audio:
            command.append('--keep-audio')
        if keep_frames:
            command.append('--keep-frames')
        if map_faces:
            command.append('--map-faces')
        if many_faces:
            command.append('--many-faces')

        print(command)

        # Call the processing script
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the command was executed successfully
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Processing failed: {result.stderr}")

        # Verify if output file exists
        if not os.path.exists(output):
            raise HTTPException(status_code=500, detail="Output file not generated")

        # Schedule the cleanup task for the output file
        # background_tasks.add_task(delete_file, output)

        # Return the output file
        return FileResponse(path=output, filename=os.path.basename(output))

    finally:
        # Clean up the uploaded files after processing
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(target_path):
            os.remove(target_path)

def delete_file(file_path: str):
    """Delete a file if it exists."""
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")

@app.get("/on_publish")
async def on_publish(request: Request):
    try:
        logger.info(f"New RTMP publish: {dict(request.query_params)}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in on_publish: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/on_play")
async def on_play(request: Request):
    try:
        logger.info(f"New RTMP play: {dict(request.query_params)}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in on_play: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/on_done")
async def on_done(request: Request):
    try:
        logger.info(f"RTMP stream ended: {dict(request.query_params)}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in on_done: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn with SSL for HTTPS
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        ssl_keyfile="./key.pem", 
        ssl_certfile="./cert.pem"
    )
