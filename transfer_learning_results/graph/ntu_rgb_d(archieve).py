import sys

sys.path.extend(['../'])
from graph import tools

#for kinect v2
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                   (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                   (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                   (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

#for kinect v1
# num_node = 20
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
#                     (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
#                     (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
#                     (19, 18)]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
