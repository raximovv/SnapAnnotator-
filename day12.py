import base64
import io
import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

MODEL = "moondream"
OLLAMA_URL = "http://localhost:11434/api/generate"
WINDOW_NAME = "SnapAnnotator"
MAX_IMAGE_SIZE = 512
MAX_OBJECTS = 8

ANALYZE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["description", "objects"],
}

CLICK_TARGETS: List[Tuple[Tuple[int, int, int, int], str]] = []
PENDING_CLICK_OBJECT: Optional[str] = None


def check_setup() -> None:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        print("ERROR: Ollama is not installed or not on PATH.")
        print("Install Ollama, then run: ollama serve")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: Ollama is taking too long to respond.")
        print("Make sure the server is running with: ollama serve")
        sys.exit(1)

    if result.returncode != 0:
        print("ERROR: Ollama is not responding.")
        print("Run: ollama serve")
        sys.exit(1)

    if MODEL.lower() not in result.stdout.lower():
        print(f"ERROR: Model '{MODEL}' was not found.")
        print(f"Fix: ollama pull {MODEL}")
        sys.exit(1)


def bgr_frame_to_base64(frame_bgr) -> str:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIZE / float(max(h, w)))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def post_ollama(payload: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start it with 'ollama serve'."
        ) from exc


ANALYZE_PROMPT = textwrap.dedent(
    f"""
    You are analyzing a webcam snapshot.

    Return ONLY valid JSON that matches this schema exactly:
    {json.dumps(ANALYZE_SCHEMA)}

    Rules:
    - description: 1 concise sentence about the scene
    - objects: list the most visible physical objects only
    - max {MAX_OBJECTS} objects
    - each object name should be short, like \"laptop\" or \"coffee mug\"
    - confidence is optional, from 0.0 to 1.0
    - notes is optional and very short
    - do not include markdown or commentary
    """
).strip()


def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def normalize_analysis(raw_response: str) -> Dict[str, Any]:
    parsed = extract_json_blob(raw_response)
    if parsed and isinstance(parsed, dict):
        description = str(parsed.get("description", "")).strip()
        raw_objects = parsed.get("objects", [])
        clean_objects: List[Dict[str, Any]] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if name:
                        clean_objects.append(
                            {
                                "name": name,
                                "confidence": item.get("confidence"),
                                "notes": str(item.get("notes", "")).strip(),
                            }
                        )
                elif isinstance(item, str) and item.strip():
                    clean_objects.append({"name": item.strip(), "confidence": None, "notes": ""})
        if description or clean_objects:
            return {
                "description": description or "Scene captured.",
                "objects": clean_objects[:MAX_OBJECTS],
            }

    # Fallback if the model ignores JSON formatting.
    lines = [line.strip(" -•\t") for line in raw_response.splitlines() if line.strip()]
    description = lines[0] if lines else "Scene captured."
    objects: List[Dict[str, Any]] = []
    for line in lines[1:1 + MAX_OBJECTS]:
        line = line.lstrip("0123456789. ")
        if line:
            objects.append({"name": line, "confidence": None, "notes": ""})
    return {"description": description, "objects": objects}


def analyze_frame(frame_bgr) -> Tuple[Dict[str, Any], float, str]:
    img_b64 = bgr_frame_to_base64(frame_bgr)
    payload = {
        "model": MODEL,
        "prompt": ANALYZE_PROMPT,
        "images": [img_b64],
        "format": ANALYZE_SCHEMA,
        "stream": False,
        "keep_alive": "5m",
    }
    start = time.time()
    result = post_ollama(payload)
    elapsed = time.time() - start
    raw_text = result.get("response", "")
    return normalize_analysis(raw_text), elapsed, img_b64


def ask_followup(image_b64: str, object_name: str, question: str) -> Tuple[str, float]:
    prompt = textwrap.dedent(
        f"""
        Answer the question about the object named '{object_name}' in this image.
        Be specific to the visible object in the photo.
        If the object is not actually visible, say that clearly.
        Question: {question}
        Keep the answer to 2-4 short sentences.
        """
    ).strip()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": "5m",
    }
    start = time.time()
    result = post_ollama(payload)
    elapsed = time.time() - start
    answer = result.get("response", "").strip()
    return answer or "No answer returned.", elapsed


def wrap_lines(text: str, width: int) -> List[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def on_mouse(event, x, y, flags, param) -> None:
    global PENDING_CLICK_OBJECT
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for (x1, y1, x2, y2), object_name in CLICK_TARGETS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            PENDING_CLICK_OBJECT = object_name
            break


def build_display(
    live_frame,
    frozen_frame,
    analysis: Optional[Dict[str, Any]],
    status: str,
) -> Any:
    global CLICK_TARGETS

    frame = frozen_frame.copy() if frozen_frame is not None else live_frame.copy()
    h, w = frame.shape[:2]
    panel_w = 360
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, panel_w, cv2.BORDER_CONSTANT, value=(24, 24, 24))

    # Header
    cv2.putText(canvas, "SnapAnnotator", (w + 18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "SPACE capture | click object | q quit", (w + 18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    y = 85
    CLICK_TARGETS = []

    if analysis:
        cv2.putText(canvas, "Scene", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2)
        y += 24
        for line in wrap_lines(analysis.get("description", ""), width=34)[:4]:
            cv2.putText(canvas, line, (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
            y += 20

        y += 12
        cv2.putText(canvas, "Objects (click one)", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2)
        y += 20

        for idx, obj in enumerate(analysis.get("objects", []), start=1):
            label = obj.get("name", "object")
            conf = obj.get("confidence")
            extra = f" ({conf:.2f})" if isinstance(conf, (int, float)) else ""
            text = f"{idx}. {label}{extra}"

            top_left = (w + 18, y - 16)
            bottom_right = (w + panel_w - 20, y + 10)
            cv2.rectangle(canvas, top_left, bottom_right, (55, 55, 55), thickness=-1)
            cv2.rectangle(canvas, top_left, bottom_right, (95, 95, 95), thickness=1)
            cv2.putText(canvas, text, (w + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            CLICK_TARGETS.append(((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), label))
            y += 34
            notes = obj.get("notes", "")
            if notes:
                for note_line in wrap_lines(notes, width=30)[:2]:
                    cv2.putText(canvas, note_line, (w + 38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                    y += 16
                y += 6
    else:
        cv2.putText(canvas, "No capture yet.", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        y += 28
        cv2.putText(canvas, "Press SPACE to analyze the current frame.", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        y += 22

    # Footer/status
    footer_y = h - 40
    cv2.putText(canvas, "Status", (w + 18, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2)
    footer_y += 22
    for line in wrap_lines(status, width=34)[:4]:
        cv2.putText(canvas, line, (w + 18, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
        footer_y += 18

    return canvas


def main() -> None:
    global PENDING_CLICK_OBJECT

    check_setup()
    print("✓ Ollama + moondream ready")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No webcam found.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("\nControls:")
    print("  SPACE = capture and analyze")
    print("  Click an object in the right panel = ask a follow-up question")
    print("  q = quit\n")

    last_analysis: Optional[Dict[str, Any]] = None
    last_captured_frame = None
    last_image_b64: Optional[str] = None
    status = "Waiting for capture."

    while True:
        ret, frame = cap.read()
        if not ret:
            status = "Failed to read from webcam."
            break

        display = build_display(frame, last_captured_frame, last_analysis, status)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            last_captured_frame = frame.copy()
            status = "Analyzing captured frame..."
            display = build_display(frame, last_captured_frame, last_analysis, status)
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(1)

            try:
                analysis, elapsed, image_b64 = analyze_frame(last_captured_frame)
                last_analysis = analysis
                last_image_b64 = image_b64
                objects = [obj["name"] for obj in analysis.get("objects", [])]
                print("\n=== Capture Result ===")
                print(f"Description: {analysis.get('description', '')}")
                if objects:
                    print("Objects:")
                    for i, name in enumerate(objects, start=1):
                        print(f"  {i}. {name}")
                else:
                    print("Objects: none detected")
                status = f"Analyzed in {elapsed:.1f}s. Click an object for a follow-up question."
            except Exception as exc:  # noqa: BLE001
                status = f"Analyze failed: {exc}"
                print(f"\nERROR: {exc}\n")

        if PENDING_CLICK_OBJECT and last_image_b64:
            selected_object = PENDING_CLICK_OBJECT
            PENDING_CLICK_OBJECT = None
            print(f"\nSelected object: {selected_object}")
            print("Type a follow-up question and press Enter.")
            print("Leave it blank to use the default question.")
            try:
                user_question = input("> ").strip()
            except EOFError:
                user_question = ""

            if not user_question:
                user_question = f"What is the {selected_object} and what is it used for in this scene?"

            status = f"Asking about '{selected_object}'..."
            display = build_display(frame, last_captured_frame, last_analysis, status)
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(1)

            try:
                answer, elapsed = ask_followup(last_image_b64, selected_object, user_question)
                print(f"\nFollow-up answer ({elapsed:.1f}s):")
                print(answer)
                print()
                status = f"Answered about '{selected_object}' in {elapsed:.1f}s."
            except Exception as exc:  # noqa: BLE001
                status = f"Follow-up failed: {exc}"
                print(f"\nERROR: {exc}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("SnapAnnotator closed.")


if __name__ == "__main__":
    main()
