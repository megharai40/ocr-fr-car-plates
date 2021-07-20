import streamlit as st
import pytesseract
import cv2
import numpy as np

from PIL import Image

st.title("License Plate Recognition and Prediction ")
img = st.sidebar.file_uploader("Choose an Image: ")
if img is not None:
  img_read = Image.open(img)
  st.image(img,caption = 'Uploaded Image')

  img_array = np.array(img_read)
  cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
  img_cv = cv2.imread('out.jpg',0) # Getting Image in greyscale
  img_color = cv2.imread('out.jpg') #Getting image in color
  license_plate_model = cv2.CascadeClassifier('/content/haarcascade_russian_plate_number.xml')
  plate_number1 = license_plate_model.detectMultiScale(img_cv,1.0499,20)

  for (x,y,w,h) in plate_number1:
    roi = img_cv[y:y+h,x:x+w] #region of interest #Extracting plate 1st time
    roi_color = img_color[y:y+h,x:x+w]

  cv2.imwrite("License Plate.jpg", roi)
  cv2.imwrite("License Plate Color.jpg", roi_color) #Color image to be displayed

  plate_imgcv = cv2.imread("License Plate.jpg")

  plate_color = cv2.imread("License Plate Color.jpg")  #Color Image of Plate

  plate_number2 = license_plate_model.detectMultiScale(plate_imgcv,1.009,20) #Extracting Plate 2nd time
  for (x,y,w,h) in plate_number2:
    roi2 = plate_imgcv[y:y+h,x:x+w]
    cv2.imwrite("License Plate Final.jpg", roi2)

  plate_img_final = Image.open('License Plate Final.jpg')
  op = pytesseract.image_to_string(plate_img_final) #Extracting text

  if op == ' \n\x0c': #If text extracted is blank
    d,roi_new_binary = cv2.threshold(roi2,126,255,cv2.THRESH_BINARY) #Converting Image to Binary b/w
    cv2.imwrite("License Plate Final(binary).jpg", roi_new_binary)
    plate_img_final = Image.open('License Plate Final(binary).jpg')
    op = pytesseract.image_to_string(plate_img_final) #Extracting text from binary image

  #Resizing color image to final image
  plate_img_boxes = cv2.imread("License Plate Final.jpg")
  h_big,w_big,d_big = plate_img_boxes.shape

  boxes_bigger = cv2.resize(plate_color, (w_big,h_big))
  boxes = pytesseract.image_to_boxes(plate_img_boxes)      #image to boxes

  for b in boxes.splitlines():
    b = b.split()
    x1,y1,w1,h1 = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(boxes_bigger,(x1,h_big-y1),(w_big,h_big-h1),(0,0,255),1) 

  if st.button('PREDICT'):
    if op == ' \n\x0c':
      st.write("Plate Not Detected. Try Another JPG")
    else:
      st.write(op)
      st.image(boxes_bigger,caption = "Plate Image")
