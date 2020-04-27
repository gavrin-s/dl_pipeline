from typing import Any, Optional
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def parse_cvat_xml(fname: str, type_only: Optional[str] = None) -> pd.DataFrame:
    """
    Parse CVAT annotation from xml format:
    https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md
    :param fname: xml file name
    :param type_only: just what type to use
    :return: DataFrame with parsed annotation
    """

    def _parse_points(value: Any) -> Any:
        """
        Helper function for transform CVAT points from format `x1,y2;x2,y2;..;xn,yn`
         to format [[x1, x2], [x2, y2], ..., [xn, yn]]
        :param value: string of points. If no string is given, return the same value
        :return: points
        """
        if isinstance(value, str):
            return tuple([tuple([float(point) for point in points.split(",")])
                          for points in value.split(";")])
        return value

    with open(fname, "rb") as xml_file:
        xml = xml_file.read()

    root = etree.fromstring(xml)
    data = []
    for item in root.getchildren():
        if item.tag != "track" and item.tag != "image":
            continue
        for record in item.getchildren():
            dict_ = dict(record.attrib)
            dict_.update(dict(item.attrib))
            dict_["type"] = record.tag
            for attribute in record:
                dict_[f"attribute_{attribute.attrib['name']}"] = attribute.text

            data.append(dict_)

    df_annotation = pd.DataFrame(data)

    for int_column in ["frame", "outside", "occluded", "keyframe", "id", "width", "height"]:
        if int_column in df_annotation:
            df_annotation[int_column] = df_annotation[int_column].astype(int)

    if "box" in df_annotation["type"].values:
        df_annotation[["xtl", "xbr", "ytl", "ybr"]] = df_annotation[["xtl", "xbr", "ytl", "ybr"]].astype(float)

    if "points" in df_annotation.columns:
        df_annotation["points"] = df_annotation["points"].apply(_parse_points)

    if type_only is not None:
        df_annotation = df_annotation[df_annotation["type"] == type_only]

    return df_annotation
