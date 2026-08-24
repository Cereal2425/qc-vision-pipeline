from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import QCPipeline

app = FastAPI(title="Real-Time Automated QC & Telemetry Dashboard")
pipeline = QCPipeline(video_source="data/test_video.mp4")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves a side-by-side UI layout for the Video Feed and JSON Telemetry."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>QC Automated Telemetry Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #121212; color: #fff; margin: 0; padding: 20px; }
            h1 { text-align: center; color: #00e676; margin-bottom: 20px; }
            .container { display: flex; flex-direction: row; justify-content: center; gap: 20px; max-width: 1300px; margin: 0 auto; }
            .card { background-color: #1e1e1e; border-radius: 8px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); flex: 1; }
            img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #333; }
            pre { background-color: #000; padding: 15px; border-radius: 4px; color: #00e676; height: 400px; overflow-y: auto; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>Vision Quality Control Dashboard</h1>
        <div class="container">
            <div class="card">
                <h3>Live Video Feed</h3>
                <img src="/video-feed" alt="Real-time Stream">
            </div>
            <div class="card">
                <h3>Live Telemetry (JSON)</h3>
                <pre id="telemetry-log">Loading stream data...</pre>
            </div>
        </div>

        <script>
            // Poll telemetry data every 200ms
            async function updateTelemetry() {
                try {
                    const response = await fetch('/telemetry');
                    const data = await response.json();
                    document.getElementById('telemetry-log').textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    console.error('Telemetry Fetch Error:', err);
                }
            }
            setInterval(updateTelemetry, 200);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/telemetry")
def get_telemetry():
    _, telemetry = pipeline.process_frame()
    return telemetry

def generate_video_feed():
    while True:
        frame, _ = pipeline.process_frame()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")