from pathlib import Path

class Structure:
    def __init__(self, base_dir, batch_name):
        self.BASE = base_dir
        self.TASK = batch_name
        self.TASK_DIR = self.BASE / self.TASK
        self.DOWNLOAD_DIR = self.TASK_DIR / "downloads"
        self.COMP_DIR = self.TASK_DIR / "forge"
        self.DATA_DIR = self.TASK_DIR / "data"
        # self.LOGS_PREP_DIR = self.TASK_DIR / "data" / "prepare_logs"
        # self.LOGS_COPY_DIR = self.TASK_DIR / "data" / "copy_logs"

        for directory in [self.TASK_DIR, self.DOWNLOAD_DIR, self.COMP_DIR, self.DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # for subdir in ["success", "failed"]:
        #     (self.LOGS_COPY_DIR / subdir).mkdir(exist_ok=True)
        #     (self.LOGS_PREP_DIR / subdir).mkdir(exist_ok=True)

        # for log_file in ["success.txt", "failed.txt"]:
        #     (self.LOGS_COPY_DIR / log_file).touch(exist_ok=True)
        #     (self.LOGS_PREP_DIR / log_file).touch(exist_ok=True)

    def comp_id_dir(self, comp_id: str) -> Path:
        return self.COMP_DIR / comp_id

    def data_comp_id_dir(self, comp_id: str) -> Path:
        return self.DATA_DIR / comp_id

    def raw_dir(self, comp_id: str) -> Path:
        return self.data_comp_id_dir(comp_id) / "raw"

    def data_dir(self, comp_id: str) -> Path:
        return self.data_comp_id_dir(comp_id) / "data"

    def utils_dir(self, comp_id: str) -> Path:
        return self.data_comp_id_dir(comp_id) / "utils"
    
    # def evo_dir(self, comp_id: str) -> Path:
    #     return self.data_comp_id_dir(comp_id) / "evo"

    def public_dir(self, comp_id: str) -> Path:
        return self.data_dir(comp_id) / "public"

    def private_dir(self, comp_id: str) -> Path:
        return self.data_dir(comp_id) / "private"
