import glob
import cv2
import numpy as np

# NEEDLE
needle = 'C:/Users/Alexander/Desktop/gpu-alert/needles/newegg-needle.png'

# HAYSTACKS
haystacks = glob.glob('C:/Users/Alexander/Desktop/gpu-alert/test-images/*')

def detectNeedle(needle,haystack):

	#RESET RESULT
	result = False

	img = cv2.imread(haystack)
	scale_percent = 150
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dsize = (width, height)
	img = cv2.resize(img, dsize)
	
	#DEBUG
	'''	
	cv2.imshow("img",img)
	cv2.waitKey(0)
	'''

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	template = cv2.imread(needle, 0)

	w, h = template.shape[::-1]

	res = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED) 

	threshold = 0.95

	loc = np.where( res >= threshold)

	for pt in zip(*loc[::-1]):

		result = True

		#DEBUG
		'''
		print("Detected in " + haystack)
		cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
		cv2.imshow("img",img)
		cv2.waitKey(0)
		'''
		break

	return result

if __name__ == "__main__":
	
	for haystack in haystacks:
		
		if detectNeedle(needle,haystack) is True:
			print(haystack)
			print("Match found!")
		else:
			print(haystack)
			print("No match...")
