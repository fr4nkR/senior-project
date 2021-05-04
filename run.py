import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

BASE_PATH = "/saved"
app = Flask(__name__)
app.secret_key = b'secret'

# here change it for the actual file extension for ML models for torch
ALLOWED_EXTENSION_SCIENTIST = {'pth'}
ALLOWED_EXTENSION_OWNER = {'zip'}

def allowed_files(filename: str, allowed_extension) -> bool :
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extension

@app.route('/owner', methods=['GET', 'POST'])
def owner():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files.get('file')

        if file.filename == '':
            flash('No file was selected')
            return redirect(request.url)

        if file and allowed_files(file.filename, ALLOWED_EXTENSION_OWNER):
            filename = secure_filename(file.filename)
            file.save(os.path.join(BASE_PATH, filename))
            print(os.path.join(BASE_PATH, filename))
            flash('File uploaded successfully')
            return redirect(url_for('uploading_data'))
    return render_template('owner.html')

@app.route('/scientist', methods=['GET', 'POST'])
def scientist():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files.get('file')

        if file.filename == '':
            flash('No file was selected')
            return redirect(request.url)

        if file and allowed_files(file.filename, ALLOWED_EXTENSION_SCIENTIST):
            filename = secure_filename(file.filename)
            file.save(os.path.join(BASE_PATH, filename))
            print(os.path.join(BASE_PATH, filename))
            flash('File uploaded successfully')
            return redirect(url_for('uploading_model'))
    return render_template('scientist.html')

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def decision():
    return render_template('decision.html')

@app.route('/uploading_data', methods=['GET'])
def uploading_data():
    return render_template('uploading_data.html')

@app.route('/uploading_model', methods=['GET'])
def uploading_model():
    return render_template('uploading_model.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

