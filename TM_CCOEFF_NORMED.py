import cv2

input_image = cv2.imread('virat.jpg')
template_image = cv2.imread('viratcp.jpg')
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
# Perform template matching using TM_CCOEFF_NORMED method
result = cv2.matchTemplate(input_gray, template_gray, cv2.TM_CCOEFF_NORMED)
# Get the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# Draw a rectangle around the best match
template_width, template_height = template_gray.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
cv2.rectangle(input_image, top_left, bottom_right, (0, 0, 255), 2)

cv2.imshow('Result', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()