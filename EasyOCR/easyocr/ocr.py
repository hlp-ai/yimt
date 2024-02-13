import easyocr

reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('./examples/chinese.jpg')
print(result)