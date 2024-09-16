#!/usr/bin/env python3
"""This script, which should be run from the `docs/` directory, updates all the
notebooks under `docs/tutorials` in order to automatically perform mass updates
of certain common cells (header, footer, install steps).

Authors

* Sylvain de Langen 2024"""

import glob
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

with open("tutorials/notebook-header.md", "r", encoding="utf-8") as header_file:
    HEADER_CONTENTS = header_file.read()


def find_first_cell_with_tag(
    cell_list: list, tag_to_find: str
) -> Optional[dict]:
    """Returns the first cell to match a given tag given a jupyter cell dict.

    Arguments
    ---------
    cell_list: list
        List of cells from a notebook as loaded from JSON
    tag_to_find: str
        The tag to look up inside of `cell["metadata"]["tags"]`

    Returns
    -------
    dict
        First entry in the `cell_list` which has the matching tag as part of the
        metadata. If none match, then `None` is returned.
    """

    for cell in cell_list:
        tags = cell.get("metadata", {}).get("tags", {})
        if tag_to_find in tags:
            return cell

    return None


def update_header_cell(header_cell: dict, tutorial_path: str):
    """Updates the content of a jupyter cell from the header template.

    Arguments
    ---------
    header_cell: dict
        Header cell in dict format as loaded from JSON
    tutorial_path: str
        Path to the tutorial, to substitute `{tutorialpath}` in the markdown
        template
    """

    header_cell.update(
        {
            "cell_type": "markdown",
            "metadata": {"id": "sb_auto_header", "tags": ["sb_auto_header"]},
            "source": HEADER_CONTENTS.replace(
                "{tutorialpath}", tutorial_path
            ).splitlines(True),
        }
    )


def update_notebook(fname: str):
    """Updates the tagged programmatically updated cells, see:

    - `tutorials/notebook-header.md`

    These cells are created when they don't exist, but they can be moved around
    as they are identified by their cell tag.

    Arguments
    ---------
    fname: str
        Path to the notebook to update."""

    logging.info(f"Updating {fname}")

    tutorial_path = fname.replace("./", "")

    with open(fname) as f:
        nb = json.load(f)

        cells = nb["cells"]
        header_cell = find_first_cell_with_tag(cells, "sb_auto_header")
        if header_cell is None:
            logging.info("Header not found; creating")
            cells.insert(0, {})
            header_cell = cells[0]

        update_header_cell(header_cell, tutorial_path)

    with open(fname, "w") as wf:
        json.dump(nb, wf, indent=1, ensure_ascii=False)
        print(file=wf)  # print final newline that jupyter adds apparently


if __name__ == "__main__":
    for fname in glob.glob("./tutorials/**/*.ipynb", recursive=True):
        update_notebook(fname)