import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import torch

from processing import NormalizeBbox
from pklot_dataset import PkLotDataset


MAX_ITERS = 1000
STOP_TRESHOLD = 1e-9
IMAGE_SIZE = (1280, 720)
INPUT_SIZE = (416, 416)
ANCHOR_COUNT = 9


torch.manual_seed(0)

ds = PkLotDataset("./data/pklot", ann_transform=NormalizeBbox(IMAGE_SIZE), load_images=False)
bbox_sizes = torch.cat([
    (annotations * torch.tensor([0, 0, 0, INPUT_SIZE[0], INPUT_SIZE[1]]))[..., 3:]
    for _, annotations, _, _
    in ds
])
bbox_sizes = bbox_sizes[((0 <= bbox_sizes) * (bbox_sizes < INPUT_SIZE[0])).all(dim=-1)]  # filter out invalid data

anchors = torch.zeros(ANCHOR_COUNT, 2)
for j in range(ANCHOR_COUNT):
    random_bbox = bbox_sizes[torch.randint(0, len(bbox_sizes), (1, ))]
    while len(anchors[(anchors == random_bbox).all(dim=-1)]) > 0:
        random_bbox = bbox_sizes[torch.randint(0, len(bbox_sizes), (1, ))]
    anchors[j] = random_bbox
print('Starting anchors:', anchors)
distances = torch.zeros(len(bbox_sizes), ANCHOR_COUNT)

i = 0
point_groups = []
for i in range(MAX_ITERS):
    print(f'Iteration {i}...')
    for j in range(ANCHOR_COUNT):
        distances[..., j] = torch.linalg.vector_norm(bbox_sizes - anchors[j], dim=-1)
    # print([distances[(bbox_sizes == anchors[j]).all(dim=-1)][:10] for j in range(ANCHOR_COUNT)])
    nearest_anchors = distances.min(dim=1).indices
    # print(nearest_anchors[:10])
    point_groups = [
        bbox_sizes[nearest_anchors == j]
        for j
        in range(ANCHOR_COUNT)
    ]
    for j, group in enumerate(point_groups):
        if len(group) == 0:
            print(f'No nearby points for group {j}, aborting...')
            exit()
    # print([len(group) for group in point_groups])
    new_anchors = torch.stack([
        group.mean(dim=0)
        for group
        in point_groups
    ])
    sorted_indices = (new_anchors[..., 0] + new_anchors[..., 1] / INPUT_SIZE[1]).sort(dim=0).indices
    # print(new_anchors)
    # print(sorted_indices)
    new_anchors = new_anchors[sorted_indices]
    # print(new_anchors)
    if torch.linalg.vector_norm(anchors - new_anchors) < STOP_TRESHOLD:
        break
    anchors = new_anchors
    if anchors.isnan().any():
        raise ValueError('NaNs detected')
rounded_anchors = anchors.round().int()

print(f'Ended after {i + 1} iterations')
print(rounded_anchors.tolist())

for j in range(ANCHOR_COUNT):
    plt.scatter(point_groups[j][..., 0], point_groups[j][..., 1], 24)
plt.scatter(rounded_anchors[..., 0], rounded_anchors[..., 1], 72, 'black', marker=MarkerStyle('X'))
plt.scatter(rounded_anchors[..., 0], rounded_anchors[..., 1], 24, 'white', marker=MarkerStyle('X'))
plt.show()
