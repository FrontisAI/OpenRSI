# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
import tarfile
import zipfile
import py7zr
import os
import shutil

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).parent.parent


def tar_directory(root_dir, output_file):
    logger.info(f"Creating tarball from {root_dir} to {output_file}")
    with tarfile.open(output_file, "w") as tar:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=os.path.relpath(file_path, os.path.dirname(root_dir)))


def get_competition_ids_in_split(split_id):
    fname = Path(__file__).parent.parent / "splits" / f"{split_id}.txt"

    with open(fname, "r") as f:
        competition_ids = f.read().splitlines()

    return competition_ids


def extract_all_from_path(
    path: Path, already_extracted: set = set(), force: bool = True, delete_compressed: bool = False
) -> None:
    """Extracts the contents of a compressed file to a destination directory."""

    def extract(file: Path, dst: Path) -> None:
        """Extracts a compressed file to the specified destination."""
        # Check if the destination directory already exists
        if dst.name.endswith(".json") or dst.name.endswith(".csv") or dst.name.endswith(".txt"):
            logger.info(f"Adjusting destination from {dst} to {dst.parent}")
            dst = dst.parent
            
        try:
            if file.suffix == ".7z":
                with py7zr.SevenZipFile(file, mode="r") as ref:
                    ref.extractall(dst)
            elif file.suffix == ".zip":
                with zipfile.ZipFile(file, "r") as ref:
                    ref.extractall(dst)
            elif file.suffix == ".gz" or file.name.endswith(".tar.gz"):
                with tarfile.open(file, "r:gz") as ref:
                    ref.extractall(dst)
            else:
                raise NotImplementedError(f"Unsupported compression format: `{file.suffix}`.")
        except Exception as e:
            # final attempt to extract using shutil
            logger.error(f"Failed to extract {file} with suffix {file.suffix}: {e}; will try with shutil.")
            try:
                shutil.unpack_archive(file, dst)
            except Exception as e2:
                logger.error(f"Failed to extract {file} with shutil: {e2}")
                raise e2

    def is_compressed(file: Path) -> bool:
        """Check if the file is a compressed format that we can handle."""
        return file.suffix in {".7z", ".zip", ".gz"} or file.name.endswith(".tar.gz")

    if is_compressed(path):
        if path.name.endswith(".tar.gz"):
            dst = path.parent / path.name[:-7]  # Remove .tar.gz
        else:
            dst = path.parent / path.stem

        if dst.exists() and not force and dst not in already_extracted:
            logger.info(f"Skipping extraction for {path}, already exists at {dst}.")
        else:
            logger.info(f"Extracting {path} to {dst}")
            extract(path, dst)

        if delete_compressed:
            path_to_delete = path
            logger.info(f"Deleting compressed file {path_to_delete} after extraction.")
            path_to_delete.unlink()

        path = dst

    if not path.is_dir():
        return
    to_extract = {fpath for fpath in set(path.iterdir()) - already_extracted if is_compressed(fpath)}
    already_extracted.update(to_extract)
    for fpath in to_extract:
        extract_all_from_path(
            fpath, already_extracted=already_extracted, force=force, delete_compressed=delete_compressed
        )
