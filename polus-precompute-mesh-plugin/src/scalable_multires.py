import os, struct, json
import trimesh
from neurogen import encoder
import numpy as np
from functools import cmp_to_key
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("mesh-generation")
logger.setLevel(logging.INFO)

class Quantize():
    """
    A class used to quantize mesh vertex positions for Neuroglancer precomputed
    meshes to a specified number of bits.
    
    Based on the C++ code provided here: https://github.com/google/neuroglancer/issues/266#issuecomment-739601142

    Attributes
    ----------
    upper_bound : int 
        The largest integer used to represent a vertex position.
    scale : np.ndarray
        Array containing the scaling factors for each dimension. 
    offset : np.ndarray
        Array containing the offset values for each dimension. 
    """

    def __init__(self, fragment_origin, fragment_shape, input_origin, quantization_bits, lod):
        """
        Parameters
        ----------
        fragment_origin : np.ndarray
            Minimum input vertex position to represent.
        fragment_shape : np.ndarray
            The inclusive maximum vertex position to represent is `fragment_origin + fragment_shape`.
        input_origin : np.ndarray
            The offset to add to input vertices before quantizing them.
        quantization_bits : int
            The number of bits to use for quantization.
        """

        self.upper_bound = np.iinfo(np.uint32).max >> (np.dtype(np.uint32).itemsize*8 - quantization_bits) # if 10 then 1023, if 16 then 65535
        self.scale = np.floor(self.upper_bound / fragment_shape)
        self.offset = input_origin - fragment_origin + (fragment_shape/(2**(lod)))/self.scale
    
    def __call__(self, vertices):
        """ Quantizes an Nx3 numpy array of vertex positions.
        
        Parameters
        ----------
        vertices : np.ndarray
            Nx3 numpy array of vertex positions.
        
        Returns
        -------
        np.ndarray
            Quantized vertex positions.
        """
        output = np.minimum(self.upper_bound, np.maximum(0, self.scale*(vertices + self.offset))).astype(np.uint32)
        return output
 

def cmp_zorder(lhs, rhs):
    """Compare z-ordering
    
    Code taken from https://en.wikipedia.org/wiki/Z-order_curve
    """
    def less_msb(x: int, y: int):
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


def generate_mesh_decomposition(mesh, nodes_per_dim, quantization_bits, nodearray, frag, lod, maxvertex, minvertex):
    """Decomposes and quantizes a mesh according to the desired number of nodes and bits.
    
    A mesh is decomposed into a set of submeshes by partitioning the bounding box into
    nodes_per_dim**3 equal subvolumes . The positions of the vertices within 
    each subvolume are quantized according to the number of bits specified. The nodes 
    and corresponding submeshes are sorted along a z-curve.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    nodes_per_dim : int
        Number of nodes along each dimension.
    quantization_bits : int
        Number of bits for quantization. Should be 10 or 16.
    
    Returns
    -------
    nodes : list
        List of z-curve sorted node coordinates corresponding to each subvolume. 
    submeshes : list
        List of z-curve sorted meshes.
    """

    # Scale our mesh coordinates.
    maxvertex = mesh.vertices.max(axis=0)
    minvertex = mesh.vertices.min(axis=0)
    nodearray = nodearray
    scale = nodearray/(maxvertex- minvertex)
    verts_scaled = scale*(mesh.vertices - minvertex) #the scaled vertices ranges from 0 to chunk_shape
    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = verts_scaled

    # Define plane normals and scale mesh.
    nyz, nxz, nxy = np.eye(3)
    res = [i*j for i,j in zip([1,1,1], nodearray)]
    # create submeshes. 
    submeshes = []
    nodes = []
    for x in range(0, nodearray[0]):
        mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
        for y in range(0, nodearray[1]):
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
            for z in range(0, nodearray[2]):
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
                # Initialize Quantizer.
                quantizer = Quantize(
                    fragment_origin=np.array([x, y, z]), 
                    fragment_shape=np.array(frag), 
                    input_origin=np.array([0,0,0]), 
                    quantization_bits=quantization_bits,
                    lod = lod
                )
    
                if len(mesh_z.vertices) > 0:
                    mesh_z.vertices = quantizer(mesh_z.vertices)
                    submeshes.append(mesh_z)
                    nodes.append([x,y,z])
    
    # Sort in Z-curve order
    submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))
            
    return nodes, submeshes

def generate_trimesh_chunks(
    mesh,
    directory,
    segment_id,
    chunks):
    """Generates temporary chunks of the meshes by saving them in ply files

    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    directory : str
        Temporary directory to save the ply files
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    chunks: tuple
        The X, Y, Z chunk that is analyzed
    """
    chunk_filename = '{}_{}_{}_{}.ply'.format(segment_id, chunks[0], chunks[1], chunks[2])
    temp_dir = os.path.join(directory, "temp_drc")
    os.makedirs(temp_dir, exist_ok=True)
    mesh.export(os.path.join(temp_dir, chunk_filename))

def generate_multires_mesh(
    mesh, 
    directory, 
    segment_id, 
    quantization_bits=16,
    compression_level=4,
    mesh_subdirectory='meshdir'):
    
    """ Generates a Neuroglancer precomputed multiresolution mesh.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    directory : str
        Neuroglancer precomputed volume directory.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    chunks: tuple
        The X, Y, Z chunk that is analyzed
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.    
    """

    def solve_for_nodes_per_dim(num_lods):
        """This function solves for which dimension to slice with each progressive mesh
            The number of nodes each dimension is either one (no slicing) or two (slicing).
            If the number of nodes for all dimensions is two, then you have a full octree.

        Parameters
        ----------
        num_lods : int 
            The number of progressive meshes
        """
        nodes_per_dim = []
        fragment_shapes = []
        if num_lods > 0: 
            for i in range(0, num_lods):
                if i == 0:
                    nodes_per_dim.append([1, 1, 1]) 
                else:
                    maxval = np.max(nodes)
                    newnodes = []
                    for node in range(3):
                        if nodes[node] == maxval:
                            nodes[node] = nodes[node] - 1
                            newnodes.append(int(nodes_per_dim[-1][node]*2))
                        else:
                            newnodes.append(int(nodes_per_dim[-1][node]))
                    nodes_per_dim.append(newnodes)
            lastnode = nodes_per_dim[-1]
            for i in range(num_lods):
                append = [item*(2**(num_lods-i-1))for item in nodes_per_dim[i]] 
                append = [int(i/j) for i, j in zip(append, lastnode)]
                fragment_shapes.append(append)
        else: # if there are no progressive meshes to create
            num_lods = 1
            fragment_shapes = [[1,1,1]]
            nodes_per_dim = [[1,1,1]]

        return num_lods, nodes_per_dim, fragment_shapes

    dim_goal = (7, 7, 7) # (The smallest possible chunk size is (2^7, 2^7, 2^7))
    bounds = mesh.bounds
    maxvertex = bounds[1]
    minvertex = bounds[0]
    grid_origin = minvertex
    shape = bounds[1] - bounds[0]

    # Need to solve for the number of progressive meshes to meet dimension goal. 
    nodes = np.floor(np.log2(shape)) # convert shape to log base 2
    num_lods = int(np.max(nodes - dim_goal)) + 1

    num_lods, nodes_per_dim, fragment_shapes = solve_for_nodes_per_dim(num_lods)

    # Define key variables. 
    lods = np.arange(0, num_lods)

    chunk_shape = shape/(nodes_per_dim[-1])

    lod_scales = np.array([2**lod for lod in lods])
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])

    # Clean up mesh.
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fill_holes()

    # Create directory
    mesh_dir = os.path.join(directory, mesh_subdirectory)
    os.makedirs(mesh_dir, exist_ok=True)

    fragment_offsets = []
    fragment_positions = []
    # Write fragment binaries.
    with open(os.path.join(mesh_dir, f'{segment_id}'), 'wb') as f:
        ## We create scales from finest to coarsest.
        i = num_lods
        for scale in lod_scales[::-1]: #go in scale backwards 
            num_faces = int(mesh.faces.shape[0]//(lod_scales.max()/scale)**2)
            scaled_mesh = mesh.simplify_quadratic_decimation(num_faces)
            scaled_mesh.remove_degenerate_faces()
            scaled_mesh.remove_duplicate_faces()
            scaled_mesh.remove_unreferenced_vertices()
            scaled_mesh.remove_infinite_values()
            scaled_mesh.fill_holes()

            nodearray = nodes_per_dim[i-1]
            frag = [int(x) for x in fragment_shapes[i-1]]
            nodes, submeshes = generate_mesh_decomposition(scaled_mesh, scale, quantization_bits, nodearray, frag, i, maxvertex, minvertex)
            lod_offsets = []
            for submesh in submeshes:
                # Only write non-empty meshes.
                if len(submesh.vertices) > 0:
                    draco = encoder.encode_mesh(submesh,compression_level=compression_level)
                    f.write(draco)
                    lod_offsets.append(len(draco))
                else:
                    lod_offsets.append(0)

            fragment_positions.append(np.array(nodes))
            fragment_offsets.append(np.array(lod_offsets))
            i = i - 1
    
    num_fragments_per_lod = np.array([len(nodes) for nodes in fragment_positions])

    # Write manifest file.
    with open(os.path.join(mesh_dir, f'{segment_id}.index'), 'wb') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())
        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))
        f.write(num_fragments_per_lod.astype('<I').tobytes())
        for frag_pos, frag_offset in zip(fragment_positions, fragment_offsets):
            f.write(frag_pos.T.astype('<I').tobytes(order='C'))
            f.write(frag_offset.astype('<I').tobytes(order='C'))
    