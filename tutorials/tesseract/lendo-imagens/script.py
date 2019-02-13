import pytesseract as ocr
import numpy as np
import cv2

from PIL import Image


# imagem = Image.open('saoluis.jpg').convert('RGB')
imagem = cv2.imread('saoluis.jpg')

height, width, _depth = imagem.shape
imagem = imagem[:, 0:int(width / 2)]
#cv2.imshow('',imagem)
#cv2.waitKey(0)

npimagem = np.asarray(imagem).astype(np.uint8)

npimagem[:, :, 0] = 0 # zerando o canal R (RED)
npimagem[:, :, 2] = 0 # zerando o canal B (BLUE)
teste = Image.fromarray(npimagem)#.show()
#cv2.waitKey(0)

im = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
# cv2.imshow('', im)
# cv2.waitKey(0)

#ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(im, 150, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
# ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binimagem = Image.fromarray(thresh)
binimagem.show()
cv2.waitKey(0)

phrase = ocr.image_to_string(binimagem, lang='por')

print(phrase)
