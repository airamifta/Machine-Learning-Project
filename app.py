import os
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import tensorflow as tf

# Menyembunyikan pesan log TensorFlow untuk membuat output lebih bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Membuat instance aplikasi Flask
app = Flask(__name__)
# Menentukan folder untuk mengunggah file
UPLOAD_FOLDER = 'uploads'
# Menentukan ekstensi file yang diperbolehkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Mengkonfigurasi aplikasi Flask dengan folder unggahan
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mengatur TensorFlow untuk menggunakan perangkat yang tersedia secara otomatis
tf.config.set_soft_device_placement(True)

# Memuat model TFLite dari file
interpreter = tf.lite.Interpreter(model_path='ai_model/model3.tflite')
# Mengalokasikan memori untuk tensor input dan output
interpreter.allocate_tensors()

# Mendapatkan detail tensor input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi untuk memeriksa apakah file diperbolehkan berdasarkan ekstensi
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk memproses dan mengaugmentasi gambar sebelum diprediksi
def preprocess_and_augment_image(image_path, target_size=(224, 224)):
    # Memuat gambar dengan ukuran target
    image = load_img(image_path, target_size=target_size)
    # Mengkonversi gambar ke array
    image = img_to_array(image)
    # Menambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    return image

# Route untuk halaman utama (GET request)
@app.route('/', methods=['GET'])
def index():
    # Merender template HTML 'index.html'
    return render_template('index.html')

# Route untuk menangani prediksi gambar (POST request)
@app.route('/', methods=['POST'])
def predict():
    # Memeriksa apakah ada file dalam request
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # Memeriksa apakah file diperbolehkan
    if file and allowed_file(file.filename):
        # Mengamankan nama file dan menyimpan ke folder unggahan
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Memproses dan mengaugmentasi gambar
        img_array = preprocess_and_augment_image(file_path)
        
        # Mengkonversi data input ke tipe float32
        input_data = np.array(img_array, dtype=np.float32)
        # Mengatur tensor input dengan data gambar
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Menjalankan inferensi
        interpreter.invoke()
        
        # Mengambil hasil prediksi dari tensor output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Mendapatkan kelas prediksi dengan nilai argmax
        predicted_class = np.argmax(output_data, axis=1)[0]
        # Mendefinisikan label kelas
        class_labels = {0: 'Bad', 1: 'Good', 2: 'Mixed'}
        
        # Merender template HTML 'predict.html' dengan hasil prediksi
        return render_template('predict.html', context={
            'result': class_labels[predicted_class],
            'img': file_path,
        })

    # Mengembalikan pesan jika file tidak diperbolehkan
    return 'File not allowed'

# Route untuk mengunduh file yang diunggah
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Membuat folder unggahan jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Menjalankan aplikasi Flask pada host dan port yang ditentukan
app.run(debug=False, host='0.0.0.0', port=8000)
