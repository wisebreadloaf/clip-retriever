from extractor import extract_images_from_all_pdfs
from clip_inference.main import main
from index import run_clip_index
from filter import clip_filter

def run_program():
   extract_images_from_all_pdfs("./source_documents", "./source_images")
   print("images extracted")

   input_dataset = './source_images'
   output_folder = './output_folder'
   input_format = "files"
   cache_path = None
   batch_size = 256
   num_prepro_workers = 8
   enable_text = True
   enable_image = True
   enable_metadata = False
   write_batch_size = 10**6
   wds_image_key = "jpg"
   wds_caption_key = "txt"
   clip_model = "ViT-B/32"
   mclip_model = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
   use_mclip = False
   use_jit = False
   distribution_strategy = "sequential"
   wds_number_file_per_input_file = 10000
   output_partition_count = None
   wandb_project = "clip_retrieval"
   enable_wandb = False
   clip_cache_path = None
   slurm_job_name = None
   slurm_partition = None
   slurm_nodes = None
   slurm_job_comment = None
   slurm_nodelist = None
   slurm_exclude = None
   slurm_job_timeout = None
   slurm_cache_path = None
   slurm_verbose_wait = False

   main(
    input_dataset,
    output_folder,
    input_format,
    cache_path,
    batch_size,
    num_prepro_workers,
    enable_text,
    enable_image,
    enable_metadata,
    write_batch_size,
    wds_image_key,
    wds_caption_key,
    clip_model,
    mclip_model,
    use_mclip,
    use_jit,
    distribution_strategy,
    wds_number_file_per_input_file,
    output_partition_count,
    wandb_project,
    enable_wandb,
    clip_cache_path,
    slurm_job_name,
    slurm_partition,
    slurm_nodes,
    slurm_job_comment,
    slurm_nodelist,
    slurm_exclude,
    slurm_job_timeout,
    slurm_cache_path,
    slurm_verbose_wait,
   )
   print("")
   print("")
   print("inference done")

   embeddings_folder = './source_images'
   index_folder = './index_folder'
   max_index_memory_usage = "4G"
   current_memory_available = "16G"
   copy_metadata = True
   image_subfolder = "img_emb"
   text_subfolder = "text_emb"
   nb_cores = None

   run_clip_index("./output_folder", "./index_folder")

   print("")
   print("")
   print("indexing done")

   output_folder = './output_folder'
   indice_folder = index_folder
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
