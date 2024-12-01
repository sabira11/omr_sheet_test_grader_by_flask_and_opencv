from flask import Flask,redirect,url_for,render_template,request,flash,send_from_directory
import cv2
import os
import omr_test_grading
import omr_sheet_perspective_change  # Assuming this is your OMR processing module

app = Flask(__name__)
app.config['IMAGE_FOLDER'] = 'static/saved_image'
@app.route('/')
def index():
    return render_template('index.html', answers=None)
@app.route('/omr_perspective')
def omr_perspective():
    return render_template('omr_upload.html')
@app.route('/omr_perspective_show',methods=['POST'])
def omr_perspective_show():
    image_file = request.files["img"]
    image_path="./images/"+ image_file.filename
    image_file.save(image_path)
    processed_image = omr_sheet_perspective_change.remove_background(image_path)
    processed_filename = 'scanned_document2.jpg'
    processed_path = os.path.join(app.config['IMAGE_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, processed_image)
   
    

    return render_template('omr_upload.html',processed_image=processed_filename)
@app.route('/static/saved_image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)
   


           

if __name__ == '__main__':
    app.run(debug=True)