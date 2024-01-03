from filter import clip_filter

def run_program():
   output_folder = './output_folder'
   indice_folder = './index_folder' 
   query = 'tell me about tensorex c+'

   clip_filter(
    query,
    output_folder,
    indice_folder,
    num_results=10,
    threshold=None
   )
   print("")
   print("")
   print("filter done")

if __name__ == '__main__':
   run_program()

