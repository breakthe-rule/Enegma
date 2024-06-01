from time import time



'''
Reading PDF
'''
start = time()
import fitz  # PyMuPDF
# Open the PDF file
pdf_document = fitz.open("3_PDF-Invoice_Contractor.pdf")

# Extract text from each page
print("Reading PDF")
invoice = ''
for page_num in range(len(pdf_document)):
    page = pdf_document[page_num]
    text = page.get_text()
    if page_num>0: invoice+=text.replace("\n"," ")
    # print(f"Page {page_num+1}:\n{text}")
end = time()
print("PDF: ",end-start)



'''
Reading Images
'''
start = time()
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
img = Image.open("invoice1.jpeg")
# Extract the text from the image
print("Reading Images")
invoice = pytesseract.image_to_string(img)
print(pytesseract.image_to_data(img))
# invoice1 = invoice
# # print(invoice)
end = time()
print("IMage:",end-start)



'''
LLM
'''
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# import os
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# llm = HuggingFaceEndpoint(
#     repo_id= "mistralai/Mistral-7B-Instruct-v0.2", 
#     model_kwargs={"max_length": 1024},
#     temperature = 0.5,
#     max_new_tokens = 1024,
#     )

# prompt = PromptTemplate(
#     input_variables=["accountant"],
#     template='''You are provided with a invoice. {Accountant}'''
# )
# conversation = LLMChain(
#     llm=llm,
#     prompt=prompt,
# )

# user_message = invoice +  '''I have an extracted text from an invoice.
# Please provide the details in the following format:
# 1. "item/product/service": [list of items/products/services]
# 2. "quantity": [list of corresponding quantities]
# 3. "rate": [list of corresponding rates]
# 4. "amount": [list of corresponding amounts]
# 5. "total applied tax": number
# 6. "total discount": number
# 7. "total balance": number
# '''

# pred = conversation.invoke(input = user_message)
# print(pred["text"])



'''
Gemini
'''
import os
from langchain_google_genai import ChatGoogleGenerativeAI

print("Gemini")
os.environ["GOOGLE_API_KEY"] = "AIzaSyAPoztYy26lXfPok6y5ywOxEkJEXQtfWEI"

user_message = invoice + '''Act as  po extraction agent you need to analyse the invoice and provide neccessary fields that are required for invoice creation. The output needs to be in the below json format. 
You also need to extract all neccessary details from the invoice from the invoice and included in the output json.
{"Podate":"09-10-2023","POName":"po from usr", "Total":"$100", "LineItems":[{"Id":"item01", "price":10, "Totalprice":"$10", "Quantity":5},{"Id":"item02", "price":10, "Totalprice":"$10", "Quantity":5}]'''

llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
start = time()
result = llm.invoke(user_message)
end = time()
print(start-end)
result = result.content

result = result.strip("`").strip("json").strip().replace("\\","").replace("\n"," ")
with open("invoice_data.txt", "w") as outfile:
    outfile.write(result)



'''
Writing output in JSON
'''
import json
with open("invoice_data.json", "w") as outfile:
  json.dump(result, outfile,separators=(',', ':'))

end = time()
print("LLM:",end-start)
print("Invoice data extracted and written to invoice_data.json!")



'''
Bounding boxes around text
'''
import pytesseract
import cv2
import pandas as pd
import numpy as np
from PIL import Image

img = Image.open("invoice1.jpeg")
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)    

# perform OCR on image using pytesseract. The --psm 6 flag tells pytesseract to do it line by line
data = pytesseract.image_to_data(image,config='--oem 3 --psm 6', output_type='dict')

#this df contains the raw ocr results by pytesseract
#later we will group it together to make sensible lines
df = pd.DataFrame(data)
df = df[df["conf"] > 0]

#for each page, paragraph and line combination, create line text and bounding box dimension
page_par_line_dict = {}
master_page_par_line_list = []
master_ocr_image = ""
for index, row in df.iterrows():
    page_par_line = f"{row['par_num']}_{row['line_num']}"
    if(page_par_line not in page_par_line_dict):
        page_par_line_dict[page_par_line] = {"text": str(row["text"]) + " ", "box": (row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height'])}
    else:
        page_par_line_dict[page_par_line]["text"] = page_par_line_dict[page_par_line]["text"] + str(row["text"]) + " "
        page_par_line_dict[page_par_line]['box'] = (min(page_par_line_dict[page_par_line]['box'][0], row['left']), 
                                              min(page_par_line_dict[page_par_line]['box'][1], row['top']), 
                                              max(page_par_line_dict[page_par_line]['box'][2], row['left'] + row['width']), 
                                              max(page_par_line_dict[page_par_line]['box'][3], row['top'] + row['height']))

for entry in page_par_line_dict:
    splitted_key = entry.split('_')
    entry_value = page_par_line_dict[entry]
    master_page_par_line_list.append({
        'paragraph_number' : splitted_key[0],
        'line_number' : splitted_key[1],
        'entry_text' : entry_value['text'],
        'bounding_box' : entry_value['box']
    })

#draw bounding boxes for the lines detected in that image
for line in page_par_line_dict.values():
    if line['box'] is not None:
        cv2.rectangle(image, (line['box'][0], line['box'][1]), (line['box'][2], line['box'][3]), (0, 0, 255), 2)

if(master_ocr_image == ""):
    master_ocr_image = image
    
cv2.imwrite('master_ocr_image.jpg', master_ocr_image)

#master ocr df with all pages, paragraph, lines, text and bounding box info
master_ocr_df = pd.DataFrame(master_page_par_line_list)
print(master_ocr_df.head())