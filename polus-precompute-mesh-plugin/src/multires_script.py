import numpy as np
import open3d as o3d
import dlib
from pathlib import Path
from skimage import measure

import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS
import bioformats
import javabridge as jutil

import struct
import traceback
import math

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    num_fragments_per_lod = []
    fragment_positions = []
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i) #the number of triangles = number of fragments
        axisaligned = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh_lod)
        boxpoints = o3d.geometry.AxisAlignedBoundingBox.get_box_points(axisaligned)
        nparr = np.asarray(boxpoints)
        print("LOD", i)
        print(nparr)
        # print(mesh_lod)
        meshfile = str(path)+"lod_"+str(i)+extension
        fragment_position = np.asarray(mesh_lod.triangles) #list of indexes of where the triangle indices are
        # print("Fragment shape", fragment_position.shape)
        num_fragments_per_lod.append(fragment_position.shape[0])
        fragment_positions.append(fragment_position.astype('uint32'))
        # print(fragment_positions)
        o3d.io.write_triangle_mesh(meshfile, mesh_lod)
        mesh_lods[i]=mesh_lod
    num_fragments_per_lod = np.asarray(num_fragments_per_lod)
    # print(mesh_lods)
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods, num_fragments_per_lod, fragment_positions


input_path = Path('/home/ec2-user/3dtest')
dataname = input_path.joinpath('dA30_5_dA30.Labels.ome.tif')


output_path = Path('/home/ec2-user/polusNEUROGLANCER/multi-resolution/polus-precompute-mesh-plugin/src/multi/')
# output_path.mkdir(exist_ok=True)


log_config = Path(__file__).parent.joinpath("log4j.properties")
jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
try:
    br = BioReader(str(dataname))
    volume = br.read_image().squeeze()
    IDS = np.unique(volume)
    # need to create a for loop for all the ids.
    # for iden in IDS:
    iden = 5
    vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==IDS[iden], step_size=1)
    
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    normalize_v3(n)
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    normalize_v3(norm)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.normals = o3d.utility.Vector3dVector(norm)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = (np.mean(distances))
    radius = (3 * avg_dist)

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius*2]))
    numtri = 32
    octree = o3d.geometry.Octree.convert_from_point_cloud(bpa_mesh, point_loud=pcd, size_expand=0.01)
    print("Octree")
    print(octree)
    print(octree.size)
    # axisaligned = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(bpa_mesh)
    # print("Axis Aligned")
    # print("dimension", o3d.geometry.AxisAlignedBoundingBox.dimension(axisaligned))
    # boxpoints = o3d.geometry.AxisAlignedBoundingBox.get_box_points(axisaligned)
    # nparr = np.asarray(boxpoints)
    # print(nparr)

    # dec_mesh = bpa_mesh.simplify_quadric_decimation(numtri)
    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()

    # o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", dec_mesh)
    # o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)

    lods = [int(numtri), int(numtri/2), int(numtri/4), int(numtri/8)]
    num_lods = len(lods)
    lod_scales = np.asarray([1, 2, 4, 8]).astype(np.float32)
    vertex_offsets = np.zeros(shape=(len(lod_scales),3))
    for row in range(len(lod_scales)):
        for col in range(3):
            vertex_offsets[row][col] = lod_scales[row]/2
    # print(vertex_offsets)

    my_lods, num_fragments_per_lod, fragment_positions = lod_mesh_export(bpa_mesh, lods, ".obj", output_path)
    # print(lods)
    fragment_offsets = []
    for item in num_fragments_per_lod:
        appendzeros = np.zeros(item)
        fragment_offsets.append(appendzeros)
    for item in fragment_positions:
        print(item)

    manifest_file = output_path.joinpath((str(iden)+".index"))
    with open(str(manifest_file), 'wb') as index:
        index.write(struct.pack("<3f",1024.0,1024.0,1024.0))
        index.write(struct.pack("<3f",0,0,0))
        index.write(struct.pack("<I",num_lods))
        index.write(lod_scales.astype('<f').tobytes(order="C"))
        index.write(vertex_offsets.astype('<f').tobytes(order="C"))
        index.write(num_fragments_per_lod.astype('<I').tobytes(order="C"))
        for i in range(num_lods):
            index.write(fragment_positions[i].astype('<I').tobytes(order="C"))
            fragoff = fragment_offsets[i]
            intlen = len(fragoff)
            for item in fragoff:
                index.write(struct.pack("<I",int(item)))
        



except Exception as e:
    jutil.kill_vm()
    traceback.print_exc()
finally:
    jutil.kill_vm()
