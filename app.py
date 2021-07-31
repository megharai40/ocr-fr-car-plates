import streamlit as st
import pytesseract
import cv2
import numpy as np

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

st.title("License Plate Recognition and Prediction ")
img = st.sidebar.file_uploader("Choose an Image: ")
if img is not None:
  img_read = Image.open(img)
  st.image(img,caption = 'Uploaded Image')

  img_array = np.array(img_read)
  cv2.imwrite('out.jpg',img_array)
  img_cv = cv2.imread('out.jpg',0) # Getting Image in greyscale
  img_color = cv2.imread('out.jpg') #Getting image in color
  license_plate_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
  plate_number1 = license_plate_model.detectMultiScale(img_cv,1.05,20)

  for (x,y,w,h) in plate_number1:
    roi = img_cv[y:y+h,x:x+w] #region of interest #Extracting plate 1st time
    roi_color = img_color[y:y+h,x:x+w]
    cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),2) #plate in box

  cv2.imwrite("License Plate.jpg", roi)
  cv2.imwrite("License Plate Color.jpg", roi_color) #Color image to be displayed

  plate_imgcv = cv2.imread("License Plate.jpg")

  plate_number2 = license_plate_model.detectMultiScale(plate_imgcv,1.018,20) #Extracting Plate 2nd time
  for (x,y,w,h) in plate_number2:
    roi2 = plate_imgcv[y:y+h,x:x+w]
    cv2.imwrite("License Plate Final.jpg", roi2)

  plate_img_final = Image.open('License Plate Final.jpg')
  op = pytesseract.image_to_string(plate_img_final)

  if op == ' \n\x0c': 
    d,roi_new_binary = cv2.threshold(roi2,126,255,cv2.THRESH_BINARY) #Converting Image to Binary b/w
    cv2.imwrite("License Plate Final(binary).jpg", roi_new_binary)
    plate_img_final = Image.open('License Plate Final(binary).jpg')
    op = pytesseract.image_to_string(plate_img_final)

  plate_display = cv2.imread("License Plate Color.jpg") #Image of Just Plate
  
  if st.button('PREDICT'):
    if op == ' \n\x0c':
      st.write("Plate Not Detected. Try Another JPG")
    else:
      st.write(op)
      st.image(img_color,caption = "Plate Image")
      st.image(plate_display)
