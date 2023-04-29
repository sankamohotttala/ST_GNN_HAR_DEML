# import tools
import numpy as np

kinect_version = 1 # either 1 or 2 ; 1=20 joints, 2=25 joints

#for kinect v2
if kinect_version == 2:
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
elif kinect_version == 1:
    num_node = 20
    self_link = [(i, i) for i in range(num_node)]
    inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
                        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
                        (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
                        (19, 18)]
    inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward
else:
    raise ValueError('kinect_version must be either 1 or 2')



def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

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
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


