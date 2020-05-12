from plyfile import *
import numpy as np
from os.path import join, dirname

"""
Author : Thibault Groueix 01.09.2019
"""


class ColorMap:
    def __init__(self):
        self.colormap_path = "auxiliary/colormap.npy"
        self.colormap = (np.load(self.colormap_path) * 255).astype('int')

    def __call__(self, index):
        """
        :param value: a float
        :return:
        """
        colors = self.colormap[index]
        return colors


def save(mesh, path, colormap):
    try:
        vertex_sources = mesh.get_attribute("vertex_sources")  # batch, nb_prim, num_point, 3
        if vertex_sources.max() > 0:
            vertex_sources = (255 * (vertex_sources / vertex_sources.max())).astype('int')
            mesh.add_attribute("vertex_red")
            mesh.add_attribute("vertex_green")
            mesh.add_attribute("vertex_blue")
            mesh.set_attribute("vertex_red", colormap.colormap[vertex_sources][:, 0])
            mesh.set_attribute("vertex_green", colormap.colormap[vertex_sources][:, 1])
            mesh.set_attribute("vertex_blue", colormap.colormap[vertex_sources][:, 2])
    except:
        pass
    vertex = [tuple(mesh.vertices[i]) for i in range(len(mesh.vertices))]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),  ('z', 'f4')])
    #print(vertex)

    face = [tuple([mesh.faces[i]]) for i in range(len(mesh.faces))]
    face = np.array(face,  dtype=[('vertex_indices', 'i4', (3,))])

    PlyData(
        [
            PlyElement.describe(vertex, 'vertex'),
            PlyElement.describe(face, 'face')
        ],
        text=True).write(path[:-3] + "ply")
