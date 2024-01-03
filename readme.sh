clip-retrieval inference ./source_images ./output_folder
clip-retrieval index ./output_folder ./index_folder
clip-retrieval filter "what is tensorex c+" ./output_folder ./index_folder --num_results 10
