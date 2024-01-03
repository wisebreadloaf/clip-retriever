import os
import fitz
from PIL import Image
import io

def extract_images_from_pdf(pdf_path, image_dir):
   pdf_file = fitz.open(pdf_path)
   for page_index in range(len(pdf_file)):
       page = pdf_file[page_index]
       image_list = page.get_images(full=True)
       for image_index, img in enumerate(image_list, start=1):
           xref = img[0]
           base_image = pdf_file.extract_image(xref)
           image_bytes = base_image["image"]
           image_ext = base_image["ext"]
           image = Image.open(io.BytesIO(image_bytes))
           image.save(os.path.join(image_dir, f"image{page_index + 1}_{image_index}.{image_ext}"))

def extract_images_from_all_pdfs(pdf_dir, image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            extract_images_from_pdf(pdf_path, image_dir)

extract_images_from_all_pdfs("./source_documents", "./source_images")
