# SnapAnnotator

SnapAnnotator is a local AI webcam tool that captures a frame, analyzes it using the moondream vision model via Ollama, and allows users to interact with detected objects.

## Features
- Press SPACE to capture a webcam frame
- AI generates a scene description
- Displays detected objects
- Click an object to ask a follow-up question

## Tech Stack
- OpenCV (webcam + UI)
- PIL (image processing)
- Ollama + moondream (vision model)
- base64 (image encoding)
