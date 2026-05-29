"""
Real-time garbage classification using your laptop webcam.
Press  Q  to quit.
"""

import cv2
import numpy as np
import tensorflow as tf

# ── Load model & labels ──────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path="models/custom_objects_int8.tflite")
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

with open("models/custom_objects_labels.txt") as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]

IMG_SZ = 96

# ── Inference helper ─────────────────────────────────────────────────────────
def run_inference(frame_bgr):
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SZ, IMG_SZ))
    arr     = resized.astype(np.float32) / 255.0
    arr     = arr[np.newaxis, :, :, np.newaxis]          # (1, 96, 96, 1)

    scale, zp = inp_det["quantization"]
    arr_int8  = np.round(arr / scale + zp).astype(np.int8)

    interpreter.set_tensor(inp_det["index"], arr_int8)
    interpreter.invoke()

    raw     = interpreter.get_tensor(out_det["index"])[0]
    o_sc, o_zp = out_det["quantization"]
    probs   = (raw.astype(np.float32) - o_zp) * o_sc
    pred    = int(np.argmax(probs))
    return CLASS_NAMES[pred], float(probs[pred]), probs

# ── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam — check device index.")

print("Webcam open. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, conf, probs = run_inference(frame)

    # Overlay top prediction
    text = f"{label}  {conf*100:.1f}%"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 80), 2, cv2.LINE_AA)

    # Mini bar chart for all classes (bottom of frame)
    h, w = frame.shape[:2]
    bar_h, bar_w = 18, 120
    for i, (cn, p) in enumerate(zip(CLASS_NAMES, probs)):
        y0 = h - (len(CLASS_NAMES) - i) * (bar_h + 4) - 10
        filled = int(bar_w * max(0.0, float(p)))
        cv2.rectangle(frame, (8, y0), (8 + bar_w, y0 + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (8, y0), (8 + filled, y0 + bar_h), (0, 200, 100), -1)
        cv2.putText(frame, f"{cn} {p*100:.0f}%", (bar_w + 14, y0 + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.imshow("Garbage Classifier  [Q = quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()