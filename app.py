from flask import Flask,redirect,url_for,render_template,request,flash
import cv2
import os
import omr_test_grading
import omr_sheet_perspective_change  # Assuming this is your OMR processing module

app = Flask(__name__)

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
   
    

    return render_template('omr_upload.html',image_path=image_path)
   


           

if __name__ == '__main__':
    app.run(debug=True)