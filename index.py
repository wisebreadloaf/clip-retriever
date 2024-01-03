import os
from autofaiss import build_index
from shutil import copytree
import logging

LOGGER = logging.getLogger(__name__)

def run_clip_index(output_folder, index_folder):
   max_index_memory_usage = "4G"
   current_memory_available = "16G"
   copy_metadata = True
   image_subfolder = "img_emb"
   text_subfolder = "text_emb"
   nb_cores = None

   def quantize(emb_folder, index_folder, index_name, max_index_memory_usage, current_memory_available, nb_cores):
       try:
           LOGGER.debug(f"starting index {index_name}")
           if os.path.exists(emb_folder):
               LOGGER.debug(
                  f"embedding path exist, building index {index_name}"
                  f"using embeddings {emb_folder} ; saving in {index_folder}"
               )
               build_index(
                  embeddings=emb_folder,
                  index_path=index_folder + "/" + index_name + ".index",
                  index_infos_path=index_folder + "/" + index_name + ".json",
                  max_index_memory_usage=max_index_memory_usage,
                  current_memory_available=current_memory_available,
                  nb_cores=nb_cores,
               )
               LOGGER.debug(f"index {index_name} done")
       except Exception as e: # pylint: disable=broad-except
           LOGGER.exception(f"index {index_name} failed")
           raise e

   def clip_index(
       embeddings_folder,
       index_folder,
       max_index_memory_usage="4G",
       current_memory_available="16G",
       copy_metadata=True,
       image_subfolder="img_emb",
       text_subfolder="text_emb",
       nb_cores=None,
   ):
       quantize(
           embeddings_folder + "/" + image_subfolder,
           index_folder,
           "image",
           max_index_memory_usage,
           current_memory_available,
           nb_cores,
       )
       quantize(
           embeddings_folder + "/" + text_subfolder,
           index_folder,
           "text",
           max_index_memory_usage,
           current_memory_available,
           nb_cores,
       )
       if copy_metadata:
           copytree(embeddings_folder + "/metadata", index_folder + "/metadata")

   clip_index(output_folder, index_folder, max_index_memory_usage, current_memory_available, copy_metadata, image_subfolder, text_subfolder, nb_cores)
