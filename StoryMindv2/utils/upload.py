from huggingface_hub import HfApi
from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo


#create_repo("ZQFive/shot_ins", token="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" ,repo_type="model")
api = HfApi(token="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")

api.upload_large_folder(folder_path="./shot_ins", repo_id="ZQFive/shot_ins",repo_type="model")