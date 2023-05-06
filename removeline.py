import cv2
import numpy as np
import matplotlib.pyplot as plt
#!pip install PyMuPDF extract image from pdf from below 
import fitz
import pathlib
desktop = pathlib.Path(".")


# Open the PDF file and get the number of pages
pdf_file = "CrossOutExample4_HQuality.pdf"
filename = pdf_file.split('.pdf')
filename = filename[0]

doc = fitz.open(pdf_file)
num_pages = doc.page_count

# Loop over each page of the PDF file
for page_num in range(num_pages):
    # Get the page object and the page pixmap
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    
    # Loop over each image on the page
    for img in page.get_images():
        # Get the image dimensions and byte data
        xref = img[0]
        width = img[1]
        height = img[2]
        image_data = doc.extract_image(xref)["image"]
        
        # Save the image to a file
        #img_file = f"page{page_num}_image{xref}.jpg"
        img_file = f"{filename}_{page_num}_image{xref}.jpg"
        with open(img_file, "wb") as f:
            f.write(image_data)
            
#list(desktop.glob("real*"))   
#print(filename)         
#print(list(desktop.glob("CrossOutExample4_HQuality_*.jpg")))
fileNameList = list(desktop.glob(f"{filename}_*.jpg"))
#print(fileNameList)

for fileName in fileNameList:
	image = cv2.imread(str(fileName))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

	kernel = np.ones((4, 2), np.uint8)
	erosion = cv2.erode(thresh, kernel, iterations=1)
	dilation = cv2.dilate(thresh, kernel, iterations=1)

	trans = dilation

	# Detect horizontal lines
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
	detect_horizontal = cv2.morphologyEx(trans, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
	cnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Remove lines in the contours
	result = image.copy()
	for c in cnts:
		# Get the bounding box of the contour
		x, y, w, h = cv2.boundingRect(c)
		# Remove the line by filling the bounding box with white color
		cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), -1)

	# Save the result
	cv2.imwrite('result_strike.jpg', result)

	# Load the OCR image
	img = cv2.imread('result_strike.jpg', cv2.IMREAD_GRAYSCALE)

	# Pre-process the image
	img = cv2.medianBlur(img, 5)
	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# Perform edge detection
	edges = cv2.Canny(img, 50, 150, apertureSize=3)

	# Find contours in the image
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Loop over the contours and check if they are strikethrough text
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		aspect_ratio = float(w) / h
		if aspect_ratio > 3 and w > 20 and h > 20:
			print("Found strikethrough text at ({}, {}) with size ({}, {})".format(x, y, w, h))

	# Display the result
	plt.imshow(result)





