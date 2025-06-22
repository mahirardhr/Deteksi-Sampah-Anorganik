from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = YOLO('weights/best.pt')

# Ubah label organik jadi custom
model.names[1] = "Tidak Terdeteksi / Bukan Sampah Anorganik"

camera = cv2.VideoCapture(0)

# Webcam stream
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model.predict(frame, conf=0.5)[0]
            if results.boxes is not None:
                boxes = results.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    conf = box.conf[0]

                    if class_id == 0:
                        label = f"Anorganik ({conf:.2f})"
                    else:
                        label = f"Tidak Terdeteksi / Bukan Sampah Anorganik ({conf:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route untuk upload gambar
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Baca gambar dan deteksi
        image = cv2.imread(filepath)
        results = model.predict(image, conf=0.5)[0]

        # Deteksi manual
        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                conf = box.conf[0]

                if class_id == 0:
                    label = f"Anorganik ({conf:.2f})"
                else:
                    label = f"Tidak Terdeteksi / Bukan Sampah Anorganik ({conf:.2f})"

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Simpan hasil
        hasil_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_deteksi.jpg')
        cv2.imwrite(hasil_path, image)

        return render_template('index.html', hasil_deteksi='hasil_deteksi.jpg')

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
