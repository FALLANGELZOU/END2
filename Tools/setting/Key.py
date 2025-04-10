
import os

# https://blog.csdn.net/zp_stu/article/details/126410323
# AutoModel.from_pretrained('bert-base-chinese', cache_dir=huggingface_cache_path)
fastTools_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
huggingface_cache_path = os.path.join(fastTools_path, ".cache", "hub")
