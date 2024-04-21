import os
from huggingface_hub import hf_hub_download
import zipfile

import warnings

SERVER_URL = os.environ.get("DB_HF_DATA", "beckhamc/design_bench_data")

# the global path to a folder that stores all data files
DATA_DIR = os.path.join(
    os.path.abspath(
    os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'design_bench_data')


class DiskResource(object):
    """A resource manager that downloads files from remote destinations
    and loads these files from the disk, used to manage remote datasets
    for offline model-based optimization problems

    Public Attributes:

    is_downloaded: bool
        a boolean indicator that specifies whether this resource file
        is present at the specified location

    disk_target: str
        a string that specifies the location on disk where the target file
        is going to be placed

    download_target: str
        a string that gives the url or the google drive file id which the
        file is going to be downloaded from

    download_method: str
        the method of downloading the target file, which supports
        "google_drive" or "direct" for direct downloads

    Public Methods:

    get_data_path():
        Get a path to the file provided as an argument if it were inside the
        local folder used for storing downloaded resource files

    download():
        Download the remote file from either google drive or a direct
        remote url and store that file at a certain disk location

    """

    @staticmethod
    def get_data_path(file_path):
        """Get a path to the file provided as an argument if it were inside
        the local folder used for storing downloaded resource files

        Arguments:

        file_path: str
            a string that specifies the location relative to the data folder
            on disk where the target file is going to be placed

        Returns:

        data_path: str
            a string that specifies the absolute location on disk where the
            target file is going to be placed

        """

        return os.path.join(DATA_DIR, file_path)

    def __init__(self, 
                 disk_target, 
                 is_absolute=True,
                 download_target=None,
                 repo_id=None,
                 download_method=None):
        """A resource manager that downloads files from remote destinations
        and loads these files from the disk, used to manage remote datasets
        for offline model-based optimization problems

        Arguments:

        disk_target: str
            a string that specifies the location on disk where the target file
            is going to be placed
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder
        download_target: str
            a string that gives the url or the google drive file id which the
            file is going to be downloaded from
        download_method: str
            the method of downloading the target file, which supports
            "google_drive" or "direct" for direct downloads

        """

        if repo_id is None:
            self.repo_id = SERVER_URL
        else:
            self.repo_id = repo_id

        self.disk_target = os.path.abspath(disk_target) \
            if is_absolute else DiskResource.get_data_path(disk_target)
        
        self.download_target = download_target
        self.download_method = download_method 
        
        #os.makedirs(os.path.dirname(self.disk_target), exist_ok=True)

    @property
    def is_downloaded(self):
        """a boolean indicator that specifies whether this resource file
        is present at the specified location

        """
        return os.path.exists(self.disk_target)

    def download(self, unzip=True):
        success = False

        if self.download_target.startswith("/"):
            download_target = self.download_target[1:]
        else:
            download_target = self.download_target

        try:
            print("repo_id={}, filename={}".format(self.repo_id,download_target))
            self.disk_target = hf_hub_download(
                repo_id=self.repo_id,
                filename=download_target,
                local_dir=DATA_DIR,
                repo_type="dataset"
            )
            success = True
        except Exception as err:
            warnings.warn(
                "Unable to download file from {}: {}. Exception: {}".format(
                    self.repo_id, download_target,
                    str(err)
                ),
                UserWarning
            )

        # unzip the file if it is zipped
        if success and unzip and self.disk_target.endswith('.zip'):
            with zipfile.ZipFile(self.disk_target, 'r') as zip_ref:
                 zip_ref.extractall(os.path.dirname(self.disk_target))

        return success
