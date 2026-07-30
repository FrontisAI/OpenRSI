# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 OpenAI
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/openai/mle-bench/blob/main/LICENSE

import argparse
import os
from pathlib import Path

from mlebench.data import (
    download_and_prepare_dataset,
)
from mlebench.registry import registry
from mlebench.utils import (
    get_logger,
)
import dojo.tasks.mlebench.utils.data as data_utils

logger = get_logger(__name__)


def main():
    parser_prepare = argparse.ArgumentParser(description="Runs agents on Kaggle competitions.")
    parser_prepare.add_argument(
        "-c",
        "--competition-id",
        help=f"ID of the competition to prepare. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "-s",
        "--split",
        help="Prepare a list of competitions specified line by line in the mlebench/splits directory.",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "--keep-raw",
        help="Keep the raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default="shared/cache/dojo/tasks/mlebench",
    )
    parser_prepare.add_argument(
        "--keep-zip",
        help="Keep the zipped raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--overwrite-checksums",
        help="[For Developers] Overwrite the checksums file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--overwrite-leaderboard",
        help="[For Developers] Overwrite the leaderboard file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--skip-verification",
        help="[For Developers] Skip the verification of the checksums.",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser_prepare.parse_args()

    new_registry = registry.set_data_dir(Path(args.data_dir))

    if args.split:
        competition_ids = data_utils.get_competition_ids_in_split(args.split)
        competitions = [new_registry.get_competition(competition_id) for competition_id in competition_ids]
    else:
        if not args.competition_id:
            parser_prepare.error("One of --competition-id or --split must be specified.")
        competitions = [new_registry.get_competition(args.competition_id)]

    for competition in competitions:
        # from mlebench.utils import (
        #     authenticate_kaggle_api,
        # )
        # api = authenticate_kaggle_api()

        # # only import when necessary; otherwise kaggle asks for API key on import
        # from kaggle.rest import ApiException

        # try:
        #     api.competition_download_files(
        #         competition=competition.id,
        #         path=".",
        #         quiet=True,
        #         force=True,
        #     )
        # except ApiException as e:
        #     if _need_to_accept_rules(str(e)):
        #         logger.warning("You must accept the competition rules before downloading the dataset.")
        #         _prompt_user_to_accept_rules(competition.id)
        #         # download_dataset(competition_id, download_dir, quiet, force)
        #     else:
        #         raise e
        # except Exception as e:
        #     print(e)

        # Support for custom prepare_fn logic
        # object.__setattr__(competition, "prepare_fn", import_fn("data.mlebench.aptos2019-blindness-detection.prepare:prepare"))
        download_and_prepare_dataset(
            competition=competition,
            keep_raw=args.keep_raw,
            overwrite_checksums=args.overwrite_checksums,
            overwrite_leaderboard=args.overwrite_leaderboard,
            skip_verification=args.skip_verification,
        )

        raw_zip_path = Path(args.data_dir).resolve() / competition.id / f"{competition.id}.zip"

        if not args.keep_zip and raw_zip_path.exists():
            os.remove(raw_zip_path)

        # Our Agents expect the data to be already extracted
        path_to_public_folder = Path(args.data_dir).resolve() / competition.id / "prepared" / "public"
        data_utils.extract_all_from_path(
            path=path_to_public_folder,
            delete_compressed=not args.keep_zip,
        )

        # tarball the public folder
        tarball_path = path_to_public_folder.parent / "public.tar"
        data_utils.tar_directory(root_dir=path_to_public_folder, output_file=tarball_path)


if __name__ == "__main__":
    main()
