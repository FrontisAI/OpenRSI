# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import multiprocessing
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Optional, Union, cast

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax

from dojo.utils.logger import get_logger

# ---------------------------
# Logger Configuration
# ---------------------------

logger = get_logger()

# ---------------------------
# Utility Functions
# ---------------------------


def get_custom_process_name():
    """Simple function to extract id and rename process"""
    current_process = multiprocessing.current_process()
    return current_process.name.split("-")[-1]


def copy_contents(source: Path, destination: Path, use_symlinks: bool = True) -> None:
    """
    Copy contents from source to destination. If the source is a file, copy the file.
    If the source is a directory, merge its contents with the destination directory.

    Args:
        source (Path): Source directory or file.
        destination (Path): Destination directory.
        use_symlinks (bool): Whether to create symbolic links instead of copying.
    """
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    if not destination.is_dir():
        raise ValueError(f"Destination {destination} must be a directory.")

    if source.is_file():
        _copy_file(source, destination, use_symlinks)
    else:
        _copy_directory(source, destination, use_symlinks)


def _copy_file(src: Path, dst_dir: Path, use_symlinks: bool) -> None:
    """
    Copy a single file to the destination directory.

    Args:
        src (Path): Source file.
        dst_dir (Path): Destination directory.
        use_symlinks (bool): Whether to create a symbolic link.
    """
    destination = dst_dir / src.name
    if destination.exists():
        logger.warning(f"File {destination} already exists. Skipping copy.")
        return

    try:
        if use_symlinks:
            destination.symlink_to(src)
            logger.debug(f"Symlinked file {src} to {destination}")
        else:
            shutil.copy2(src, destination)
            logger.debug(f"Copied file {src} to {destination}")
    except OSError as error:
        logger.error(f"Failed to {'symlink' if use_symlinks else 'copy'} file {src} to {destination}: {error}")
        if not use_symlinks:
            raise


def _copy_directory(src: Path, dst_dir: Path, use_symlinks: bool) -> None:
    """
    Merge contents of the source directory into the destination directory.

    Args:
        src (Path): Source directory.
        dst_dir (Path): Destination directory.
        use_symlinks (bool): Whether to create symbolic links.
    """
    for item in src.iterdir():
        destination = dst_dir / item.name
        if destination.exists():
            logger.warning(f"Destination {destination} already exists. Skipping.")
            continue

        try:
            if use_symlinks:
                destination.symlink_to(item)
                logger.debug(f"Symlinked {item} to {destination}")
            elif item.is_dir():
                shutil.copytree(item, destination)
                logger.debug(f"Copied directory {item} to {destination}")
            else:
                shutil.copy2(item, destination)
                logger.debug(f"Copied file {item} to {destination}")
        except OSError as error:
            logger.error(f"Failed to copy {'directory' if item.is_dir() else 'file'} {item} to {destination}: {error}")
            if item.is_dir() and not use_symlinks:
                shutil.copytree(item, destination, dirs_exist_ok=True)
                logger.debug(f"Force copied directory {item} to {destination}")


def remove_unwanted_items(path: Path, patterns: Optional[list] = None) -> None:
    """
    Remove unwanted files and directories based on specified patterns.

    Args:
        path (Path): Root directory to clean.
        patterns (Optional[list]): List of patterns to remove. Defaults to ["__MACOSX", ".DS_Store"].
    """
    if patterns is None:
        patterns = ["__MACOSX", ".DS_Store"]

    for pattern in patterns:
        for item in path.rglob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.debug(f"Removed directory {item}")
                elif item.is_file():
                    item.unlink()
                    logger.debug(f"Removed file {item}")
            except OSError as error:
                logger.error(f"Failed to remove {item}: {error}")


def extract_zip_file(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a single zip file to the specified directory.

    Args:
        zip_path (Path): Path to the zip file.
        extract_to (Path): Directory to extract contents to.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.debug(f"Extracted {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        logger.error(f"Bad zip file: {zip_path}")
        raise
    except Exception as error:
        logger.error(f"Error extracting {zip_path}: {error}")
        raise


def flatten_directory_structure(zip_output_dir: Path, expected_name: str) -> None:
    """
    Flatten the directory structure if the extracted content is nested within a single directory.

    Args:
        zip_output_dir (Path): Directory where zip was extracted.
        expected_name (str): Expected name of the single directory or file.
    """
    contents = list(zip_output_dir.iterdir())
    if len(contents) == 1 and contents[0].name == expected_name:
        single_item = contents[0]
        if single_item.is_dir():
            logger.debug(f"Flattening directory structure in {zip_output_dir}")
            for item in single_item.iterdir():
                shutil.move(str(item), zip_output_dir)
                logger.debug(f"Moved {item} to {zip_output_dir}")
            single_item.rmdir()
            logger.debug(f"Removed empty directory {single_item}")
        elif single_item.is_file():
            logger.debug(f"Renaming file {single_item} to {zip_output_dir}")
            temp_rename = zip_output_dir.with_suffix(".__tmp_rename")
            single_item.rename(temp_rename)
            zip_output_dir.rmdir()
            temp_rename.rename(zip_output_dir)
            logger.debug(f"Renamed {temp_rename} to {zip_output_dir}")


def extract_all_archives(path: Path) -> None:
    """
    Extract all zip archives within the specified path and clean up the directories.

    Args:
        path (Path): Root directory to search for zip files.
    """
    for zip_file in path.rglob("*.zip"):
        output_dir = zip_file.with_suffix("")

        if output_dir.exists():
            logger.debug(f"Skipping extraction of {zip_file} as {output_dir} already exists.")
            if output_dir.is_file() and output_dir.suffix:
                try:
                    zip_file.unlink()
                    logger.debug(f"Removed existing zip file {zip_file}")
                except OSError as error:
                    logger.error(f"Failed to remove zip file {zip_file}: {error}")
            continue

        try:
            logger.info(f"Extracting {zip_file} to {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            extract_zip_file(zip_file, output_dir)

            # Clean up unwanted files
            remove_unwanted_items(output_dir)

            # Flatten directory structure if necessary
            flatten_directory_structure(output_dir, output_dir.name)

            # Remove the zip file after successful extraction
            zip_file.unlink()
            logger.debug(f"Removed zip file {zip_file}")
        except Exception as error:
            logger.error(f"Failed to extract {zip_file}: {error}")


def get_next_log_index(directory: Path) -> int:
    """
    Determine the next available index for naming log or workspace directories.

    Args:
        directory (Path): Directory to scan for existing indices.

    Returns:
        int: Next available index.
    """
    indices = []
    for item in directory.iterdir():
        if item.is_dir():
            try:
                index = int(item.name.split("-")[0])
                indices.append(index)
            except (ValueError, IndexError):
                logger.warning(f"Ignoring directory with invalid name format: {item.name}")
    next_index = max(indices, default=-1) + 1
    logger.debug(f"Next log index in {directory}: {next_index}")
    return next_index


def prepare_configuration(cfg: OmegaConf) -> OmegaConf:
    """
    Prepare and validate the configuration.

    Args:
        cfg (OmegaConf): Initial configuration.

    Returns:
        OmegaConf: Prepared and validated configuration.

    Raises:
        ValueError: If required configuration fields are missing.
    """

    # Resolve and validate paths
    cfg.input_data_dir = _resolve_path(cfg.input_data_dir, relative_to_parent=False)
    cfg.log_dir = _resolve_path(cfg.log_dir)
    cfg.workspace_dir = _resolve_path(cfg.workspace_dir)

    # Ensure log and workspace directories exist
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Log directory: {cfg.log_dir}")
    logger.debug(f"Workspace directory: {cfg.workspace_dir}")

    # Generate a unique experiment name
    experiment_index = datetime.datetime.now().isoformat()
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{get_custom_process_name()}-{experiment_index}-{cfg.exp_name}"
    logger.info(f"Experiment name: {cfg.exp_name}")

    # Update log and workspace directories with experiment name
    cfg.log_dir = (cfg.log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (cfg.workspace_dir / cfg.exp_name).resolve()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Updated log directory: {cfg.log_dir}")
    logger.debug(f"Updated workspace directory: {cfg.workspace_dir}")

    logger.info("Configuration prepared.")

    return cast(OmegaConf, cfg)


def _resolve_path(path_str: Union[str, Path], relative_to_parent: bool = False) -> Path:
    """
    Resolve a string or Path to an absolute Path.

    Args:
        path_str (Union[str, Path]): The path to resolve.
        relative_to_parent (bool): If True, resolve relative to the parent directory of the script.

    Returns:
        Path: Resolved absolute path.
    """
    path = Path(path_str)
    if not path.is_absolute():
        if relative_to_parent:
            base_path = Path(__file__).parent.parent
            path = (base_path / path).resolve()
        else:
            path = path.resolve()
    else:
        path = path.resolve()
    logger.debug(f"Resolved path: {path}")
    return path


def display_configuration(cfg: OmegaConf) -> None:
    """
    Display the configuration in a formatted YAML style using rich.

    Args:
        cfg (OmegaConf): Configuration to display.
    """
    yaml_content = OmegaConf.to_yaml(cfg)
    syntax = Syntax(yaml_content, "yaml", theme="paraiso-dark", line_numbers=True)
    rich.print(syntax)
    logger.info("Configuration displayed.")
