import sys
import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import json
import csv

args = sys.argv[1:]

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, face_detector='sfd', device='cpu')

preds = fa.get_landmarks_from_directory(args[0])

delims =     {'face': 0,
              'eyebrow1': 17,
              'eyebrow2': 22,
              'nose': 27,
              'nostril': 31,
              'eye1': 36,
              'eye2': 42,
              'lips': 48,
              'teeth': 60
              }

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }
coords = {}
raw_coords = {}
for path in preds:
       points = {}
       raw_coord = []
       coord = {}
       pred = preds[path][-1]
       for area in pred_types.keys():
              coord[area] = {}
              pred_type = pred_types[area]
              points[area] = [pred[pred_type.slice, 0], pred[pred_type.slice, 1]]
              for i in range(len(points[area])):
                     axis = 'x' if i == 0 else 'y'
                     data = points[area][i]
                     for j in range(len(data)):
                            index = delims[area] + j
                            if not index in coord[area]:
                                   coord[area][index] = {'x': None, 'y': None}
                            if len(raw_coord) <= index:
                                   raw_coord.append([])
                            point = data[j]
                            coord[area][index][axis] = point.astype(float)
                            raw_coord[index].append(point.astype(float))
       coords[path] = coord
       raw_coords[path] = raw_coord

indexes = {
       'eye1': [37, 38, 40, 41],
       'eye2': [43, 44, 46, 47],
       'nose': [30],
       'lips1': [48],
       'lips2': [64]
       }

centroids = {}
for path in raw_coords:
       face = raw_coords[path]
       centroid = []
       for area in indexes:
              select = indexes[area]
              points = [face[i] for i in select]
              x = [p[0] for p in points]
              y = [p[1] for p in points]
              centroid.append({'x': sum(x) / len(points), 'y': sum(y) / len(points)})
       centroids[path] = centroid

# create and register new dialect
csv.register_dialect('tsv', delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
header = ['x', 'y']

for path in centroids:
       centroid = centroids[path]
       with open(path.replace('png','txt').replace('jpg','txt'), 'wt') as f:
              csv_writer = csv.DictWriter(
                        f,
                        fieldnames=header,
                        dialect='tsv', # pass the new dialect
                        extrasaction='ignore'
              )

              csv_writer.writerows(centroid)


#json_coords = json.dumps(coords, indent = 4)
#json_raw_coords = json.dumps(raw_coords, indent = 4)
#json_centroids = json.dumps(centroids, indent = 4)
#print(json_centroids)
