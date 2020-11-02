import struct,json
from pathlib import Path
from struct import *
import sys

inputpath = Path("/home/ec2-user/polusNEUROGLANCER/multi-resolution/v1.0/rois/mesh/")
# print(sys.argv[1])
# for i in range(1, 63):
print(sys.argv[1])
with open(str(inputpath.joinpath(sys.argv[1])), 'rb') as file:
    file.seek(0,2)
    eof = file.tell()
    file.seek(0,0)
    print("Total number of bytes in file: ", eof)
    print(" ")

    chunkshapes = list(struct.unpack_from("<3f",file.read(12)))
    print("chunk_shape", chunkshapes)
    gridorigin = list(struct.unpack_from("<3f",file.read(12), offset=0))
    print("grid_origin", gridorigin)
    num_lods = int(struct.unpack("<I",file.read(4))[0])
    print("num_lods", num_lods)
    lod_scales = list((struct.unpack("<"+ str(num_lods)+ "f",file.read(num_lods*4))))
    print("lod_scales", lod_scales) #, lod_scales1, lod_scales2)
    for num in range(num_lods):
        vertex_offsets = list((struct.unpack("<3f", file.read(12))))
        print("vertex_offsets", num+1, ":", vertex_offsets)
    num_fragments_per_lod = list((struct.unpack("<"+ str(num_lods)+ "I",file.read(num_lods*4))))
    print("num_fragments_per_lod", num_fragments_per_lod)
    
    print("")
    print("For each lod in the range [0, num_lods): ")
    for i in range(0, num_lods):
        print("")
        # print(struct.unpack_from(str(each_numlod) + "I", file.read(numofints)))
        len_fragment_positions = int(num_fragments_per_lod[i]*3)
        len_fragment_offsets = num_fragments_per_lod[i]

        fragment_positions = list(struct.unpack_from("<" + str(len_fragment_positions) + "I", file.read(len_fragment_positions*4)))
        fragment_offsets = list(struct.unpack_from("<" + str(len_fragment_offsets)+"I", file.read(len_fragment_offsets*4)))
        print("\tfragment_positions "+ str(i+1) + ": ", fragment_positions)
        print("\tfragment_offsets "+ str(i+1) + ":   ", fragment_offsets)
        


    print(" ")
    if eof-file.tell() == 0:
        print("YOU'VE REACHED END OF FILE")
    else:
        print("YOU HAVE ", eof-file.tell(), " BYTES LEFT")

    print(" ")
    print(" ")
    # file.close()
