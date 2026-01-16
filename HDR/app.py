import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ExifTags
import base64

app = Flask(__name__)

def get_exposure_time(image_bytes, filename):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == 'ExposureTime':
                    return float(value)
    except Exception as e:
        pass
    print(f"AVISO: {filename} sem EXIF. Usando padrão.")
    return 0.033 

def array_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# --- ROTAS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if not os.path.exists('templates'):
        return jsonify({"error": "Pasta templates não encontrada"}), 500

    uploaded_files = request.files.getlist("images")
    if len(uploaded_files) < 2:
        return jsonify({"error": "Envie pelo menos 2 imagens."}), 400

    images_cv = []
    exposure_times = []

    try:
        # 1. CARREGAMENTO INICIAL
        for file in uploaded_files:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                exposure = get_exposure_time(file_bytes, file.filename)
                images_cv.append(img)
                exposure_times.append(exposure)

        if len(images_cv) < 2:
            return jsonify({"error": "Imagens inválidas."}), 400

        
        # Pega o tamanho da primeira imagem como referência
        height_ref, width_ref = images_cv[0].shape[:2]
        
        images_resized = []
        for i, img in enumerate(images_cv):
            h, w = img.shape[:2]
            # Se o tamanho for diferente, redimensiona na força bruta
            if h != height_ref or w != width_ref:
                print(f"Redimensionando imagem {i} de {w}x{h} para {width_ref}x{height_ref}")
                img = cv2.resize(img, (width_ref, height_ref))
            images_resized.append(img)
        
       
        images_cv = images_resized
       
       
        times = np.array(exposure_times, dtype=np.float32)
        if len(np.unique(times)) == 1:
             times = np.array([0.033 * (0.5**i) for i in range(len(times))], dtype=np.float32)

        # 2. ALINHAMENTO
        alignMTB = cv2.createAlignMTB()
        
        alignMTB.process(images_cv, images_cv)

        # 3. CALIBRAÇÃO
        calibrate = cv2.createCalibrateDebevec()
        response = calibrate.process(images_cv, times)

        # 4. MERGE HDR
        merge_debevec = cv2.createMergeDebevec()
        hdr_debevec = merge_debevec.process(images_cv, times, response)

        # 5. TONE MAPPING & FALSA COR
        tonemap = cv2.createTonemapDrago(gamma=2.2)
        ldr_drago = tonemap.process(hdr_debevec)
        ldr_8bit = np.clip(ldr_drago * 255, 0, 255).astype('uint8')

        gray_hdr = cv2.cvtColor(hdr_debevec, cv2.COLOR_BGR2GRAY)
        norm_img = cv2.normalize(gray_hdr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        false_color_map = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

        return jsonify({
            "hdr_preview": array_to_base64(ldr_8bit),
            "false_color": array_to_base64(false_color_map),
            "log": "Sucesso!"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)