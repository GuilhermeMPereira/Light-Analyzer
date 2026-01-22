import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ExifTags
import base64

app = Flask(__name__)

# --- FUNÇÕES AUXILIARES ---

def parse_rsp_file(file_storage):
    """Lê arquivo .rsp e converte para LUT do OpenCV"""
    try:
        content = file_storage.read().decode('utf-8').strip().split('\n')
        response_lut = np.zeros((256, 1, 3), dtype=np.float32)
        coefficients_list = []
        for line in content:
            parts = line.strip().split()
            if not parts: continue
            coeffs = [float(x) for x in parts[1:]] 
            coefficients_list.append(coeffs)
        if len(coefficients_list) < 3: return None
        coefficients_list = coefficients_list[::-1] # RGB -> BGR
        for i in range(256):
            val = i / 255.0 
            for channel in range(3):
                poly_val = sum(c * (val ** p) for p, c in enumerate(coefficients_list[channel]))
                response_lut[i, 0, channel] = poly_val
        return response_lut
    except:
        return None

def get_exposure_time(image_bytes, filename):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                if ExifTags.TAGS.get(tag_id, tag_id) == 'ExposureTime':
                    return float(value)
    except: pass
    return 0.033

def apply_clahe(img):
    """Aplica CLAHE para simular remoção de Lens Flare (melhora contraste local)"""
    # Converter para LAB para operar apenas na luminosidade
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

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
    rsp_file = request.files.get("rsp_file")
    
    # Checkbox options
    opt_align = request.form.get('auto_align') == 'true'
    opt_flare = request.form.get('lens_flare') == 'true'
    opt_ghost = request.form.get('ghost_removal') == 'true'

    if len(uploaded_files) < 2:
        return jsonify({"error": "Envie pelo menos 2 imagens."}), 400

    images_cv = []
    exposure_times = []

    try:
        # 1. CARREGAMENTO E REDIMENSIONAMENTO
        for file in uploaded_files:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                exposure = get_exposure_time(file_bytes, file.filename)
                images_cv.append(img)
                exposure_times.append(exposure)

        if images_cv:
            h_ref, w_ref = images_cv[0].shape[:2]
            for i in range(len(images_cv)):
                if images_cv[i].shape[:2] != (h_ref, w_ref):
                    images_cv[i] = cv2.resize(images_cv[i], (w_ref, h_ref))

        times = np.array(exposure_times, dtype=np.float32)
        if len(np.unique(times)) == 1:
             times = np.array([0.033 * (0.5**i) for i in range(len(times))], dtype=np.float32)

        # 2. ALINHAMENTO
        if opt_align:
            print("Alinhando imagens...");
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images_cv, images_cv)
        else:
            print("Alinhamento pulado (Opção desmarcada).")

        # 3. CALIBRAÇÃO (Response Curve)
        response = None
        if rsp_file and rsp_file.filename != '':
            response = parse_rsp_file(rsp_file)
        
        if response is None:
            calibrate = cv2.createCalibrateDebevec()
            response = calibrate.process(images_cv, times)

        # 4. FUSÃO HDR (Debevec para dados físicos)
        merge_debevec = cv2.createMergeDebevec()
        hdr_debevec = merge_debevec.process(images_cv, times, response)

    
        if opt_ghost:
            print("Ghost Removal ativo: Usando fusão Mertens para preview.")
            merge_mertens = cv2.createMergeMertens()
          
            ldr_8bit = merge_mertens.process(images_cv)
            ldr_8bit = np.clip(ldr_8bit * 255, 0, 255).astype('uint8')
        else:
            tonemap = cv2.createTonemapDrago(gamma=2.2)
            ldr_drago = tonemap.process(hdr_debevec)
            ldr_8bit = np.clip(ldr_drago * 255, 0, 255).astype('uint8')

        if opt_flare:
            print("✨ Removendo Lens Flare (Aplicando CLAHE)...")
            ldr_8bit = apply_clahe(ldr_8bit)

        # 7. MAPA DE FALSA COR 
        gray_hdr = cv2.cvtColor(hdr_debevec, cv2.COLOR_BGR2GRAY)
        norm_img = cv2.normalize(gray_hdr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        false_color_map = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

        log_msg = f"Sucesso. Align: {opt_align}, Flare: {opt_flare}, Ghost: {opt_ghost}"

        return jsonify({
            "hdr_preview": array_to_base64(ldr_8bit),
            "false_color": array_to_base64(false_color_map),
            "log": log_msg
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)