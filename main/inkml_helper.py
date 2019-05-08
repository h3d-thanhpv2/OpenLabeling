import io

import cv2
import numpy as np

import xml.etree.ElementTree as etree


class InkMLHelper:
    @staticmethod
    def parse_ink(inkml_file, threshold=2):
        with io.FileIO(inkml_file) as f:
            tree = etree.parse(f)
            root = tree.getroot()

            annotation_props = None
            line_dict = {}
            for prop in root.getchildren():
                # parse main annotation properties
                if prop.tag == "annotationXML":
                    if annotation_props:
                        raise SyntaxError("More than one {} in inkml file node".format(prop.tag))
                    else:
                        annotation_props = {}
                        for p in prop.getchildren():
                            annotation_props[p.tag] = p.text

                # parse each line in file
                if prop.tag == "traceGroup":
                    line_index = prop.get("id")
                    ground_truth = None

                    for p in prop.getchildren():
                        if p.tag == "annotationXML":
                            if ground_truth is None:
                                ground_truth = p.getchildren()[0].text
                            else:
                                raise SyntaxError("Multi ground truth in single traceGroup")

                        if p.tag == "trace":
                            line = p.text

                            # TODO: using Numpy fromstring
                            coords = line.split(",")
                            # nd_coords = np.zeros(shape=(len(coords), 2), dtype=np.uint16)
                            nd_coords = []

                            for idx, coord in enumerate(coords):
                                coord = coord.rstrip().split(" ")
                                x, y = int(coord[0]), int(coord[1])

                                if idx > 0:
                                    # some points are duplicated,
                                    # so we remove them by compare dist between them to a defined threshold
                                    #                       - thanhpv
                                    euclidean_distance = np.linalg.norm(
                                        np.array((x - nd_coords[-1][0], y - nd_coords[-1][1]))
                                    )
                                    if euclidean_distance >= threshold:
                                        nd_coords.append((x, y))
                                else:
                                    nd_coords.append((x, y))

                            line_dict.setdefault(line_index, (ground_truth, []))  # if not there before
                            line_dict[line_index][1].append(np.array(nd_coords, dtype=np.uint16))

            return annotation_props, line_dict

    @staticmethod
    def draw_single_trace(group, thickness=20):
        words = []
        for _, (_, points_clusters) in group.items():

            for points_cluster in points_clusters:
                max_val = np.amax(points_cluster, axis=0)
                min_val = np.amin(points_cluster, axis=0)

                side = (max_val - min_val) + (1, 1)

                print(side)

                # if side[0] >= 8 and side[1] >= 8:
                image = np.full((side[1], side[0]), 0, dtype=np.uint8)
                for idx in range(1, len(points_cluster)):
                    start_point = points_cluster[idx] - min_val
                    end_point = points_cluster[idx - 1] - min_val

                    cv2.line(image, tuple(start_point), tuple(end_point), (255), thickness)

                # image = cv2.resize(image, None, fx=0.1, fy=0.1)
                words.append(image)

        return words

    @staticmethod
    def draw_trace_group(group, ratio=(0.1, 0.1), thickness=20):

        lines = []

        for _, (line, points_clusters) in group.items():
            # max_x = max_y = 0
            # min_x = min_y = np.inf
            max_vals = np.amax([np.amax(points_ndarr, axis=0) for points_ndarr in points_clusters], axis=0)
            min_vals = np.amin([np.amin(points_ndarr, axis=0) for points_ndarr in points_clusters], axis=0)

            # TODO: minimize this

            side = max_vals - min_vals

            # image = np.zeros(shape=(max_vals[1], max_vals[0]), dtype=np.uint8)  # FIXME: usually generates large image
            # image = np.full((max_vals[1], max_vals[0]), 255, dtype=np.uint8)

            image = np.full((side[1], side[0]), 0, dtype=np.uint8)

            for points_cluster in points_clusters:
                if len(points_cluster) > 1:
                    for idx in range(1, len(points_cluster)):
                        start_point = points_cluster[idx] - min_vals
                        end_point = points_cluster[idx - 1] - min_vals

                        cv2.line(image, tuple(start_point), tuple(end_point), (255), thickness)

            image = cv2.resize(image, None, fx=ratio[0], fy=ratio[1])

            lines.append((line, image))

        return lines
