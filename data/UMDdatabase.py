import os
import csv


class Annotation(object):
    def __init__(self, path, orientation, bbox, left_eye, right_eye, nose, left_mouth, right_mouth):
        self.path = path
        self.orientation = orientation
        self.bbox = bbox
        self.right_eye = right_eye
        self.left_eye = left_eye
        self.nose = nose
        self.left_mouth = left_mouth
        self.right_mouth = right_mouth


class UMDDatabase(object):

    def __init__(self, base_path):
        self._base_path = base_path
        self._data = {str(i): {} for i in range(1, 4)}
        self.size = 0
        for batch in self._data:
            folder = "umdfaces_batch{}".format(batch)
            annotation_file = os.path.join(self._base_path, folder, folder + "_ultraface.csv")
            with open(annotation_file, "rb") as f:
                csvreader = csv.DictReader(f, delimiter=',')
                for row in csvreader:
                    self.size += 1
                    subject = row["SUBJECT_ID"]
                    path = os.path.join(base_path, folder, row["FILE"])
                    bbox = map(float, (row["FACE_X"], row["FACE_Y"], row["FACE_WIDTH"], row["FACE_HEIGHT"]))
                    orientation = map(float, (row["ROLL"], row["PITCH"], row["YAW"]))
                    left_eye = map(float, (row["P8X"], row["P8Y"], row["VIS8"]))
                    right_eye = map(float, (row["P11X"], row["P11Y"], row["VIS11"]))
                    nose = map(float, (row["P15X"], row["P15Y"], row["VIS15"]))
                    left_mouth = map(float, (row["P18X"], row["P18Y"], row["VIS18"]))
                    right_mouth = map(float, (row["P20X"], row["P20Y"], row["VIS20"]))
                    annotation = Annotation(path, orientation, bbox, left_eye, right_eye, nose, left_mouth, right_mouth)
                    try:
                    	self._data[batch][subject].append(annotation)
                    except KeyError:
                        self._data[batch][subject] = [annotation]

    def faces(self, batches=["1", "2", "3"]):
        for b in batches:
            for subject in self._data[b]:
                for annotation in self._data[b][subject]:
                    yield annotation
                    
