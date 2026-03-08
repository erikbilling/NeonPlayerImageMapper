import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def write_eaf(
    eaf_path: Path,
    tier_id: str,
    intervals: list[tuple[int, int]],
    video_filename: str | None = None,
    fixation_annotations: list[tuple[int, int, str]] | None = None,
) -> None:
    """Build and write an ELAN (.eaf) annotation file.

    *intervals* is a list of (start_ms, end_ms) tuples.
    If *video_filename* is given it is linked as a MEDIA_DESCRIPTOR.
    """
    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "ReferenceImageMapper",
        "DATE": "",
        "FORMAT": "3.0",
        "VERSION": "3.0",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation":
            "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
    })

    header = ET.SubElement(root, "HEADER", {
        "MEDIA_FILE": "",
        "TIME_UNITS": "milliseconds",
    })
    if video_filename:
        ET.SubElement(header, "MEDIA_DESCRIPTOR", {
            "MEDIA_URL": f"file://./{video_filename}",
            "MIME_TYPE": "video/mp4",
            "RELATIVE_MEDIA_URL": f"./{video_filename}",
        })
    ET.SubElement(header, "PROPERTY", {
        "NAME": "URN",
    }).text = ""

    # Time slots
    time_order = ET.SubElement(root, "TIME_ORDER")
    ts_id = 1
    ann_id = 1

    def add_tier(
        tier_name: str,
        annotations: list[tuple[int, int, str]],
    ) -> int:
        nonlocal ts_id, ann_id

        tier = ET.SubElement(root, "TIER", {
            "LINGUISTIC_TYPE_REF": "default-lt",
            "TIER_ID": tier_name,
        })

        for start, end, label in annotations:
            ts_start = f"ts{ts_id}"
            ts_end = f"ts{ts_id + 1}"
            ET.SubElement(time_order, "TIME_SLOT", {
                "TIME_SLOT_ID": ts_start,
                "TIME_VALUE": str(start),
            })
            ET.SubElement(time_order, "TIME_SLOT", {
                "TIME_SLOT_ID": ts_end,
                "TIME_VALUE": str(end),
            })
            ts_id += 2

            ann = ET.SubElement(tier, "ANNOTATION")
            alignable = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"a{ann_id}",
                "TIME_SLOT_REF1": ts_start,
                "TIME_SLOT_REF2": ts_end,
            })
            ET.SubElement(alignable, "ANNOTATION_VALUE").text = label
            ann_id += 1

        return len(annotations)

    main_annotations = [(start, end, tier_id) for start, end in intervals]
    main_count = add_tier(tier_id, main_annotations)

    fixation_count = 0
    if fixation_annotations:
        fixation_count = add_tier("fixations", fixation_annotations)

    # Linguistic type
    ET.SubElement(root, "LINGUISTIC_TYPE", {
        "GRAPHIC_REFERENCES": "false",
        "LINGUISTIC_TYPE_ID": "default-lt",
        "TIME_ALIGNABLE": "true",
    })

    # Write file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    eaf_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(eaf_path), encoding="unicode", xml_declaration=True)
    logger.info(
        "EAF exported to %s (%d AOI + %d fixation annotations)",
        eaf_path,
        main_count,
        fixation_count,
    )
