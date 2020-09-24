import logging, argparse, bioformats
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import utils
import filepattern
from filepattern import FilePattern as fp
import os
import itertools
import numpy as np
from multiprocessing import cpu_count


if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)
    parser.add_argument('--imageNum', dest='image_num', type=str,
                        help='Image number, will be stored as a timeframe', required=True)
    parser.add_argument('--stackheight', dest='stack_height', type=int,
                        help='The Height of the Stack', required=True)
    parser.add_argument('--stackby', dest='stack_by', type=str,
                        help='Variable that the images get stacked by', required=True)
    parser.add_argument('--varsinstack', dest='vars_instack', type=str,
                        help='Variables that the stack shares', required=True)
    parser.add_argument('--valinstack', dest='vals_instack', type=int, nargs='+',
                        help='Values of variables that the stack shares', required=True)
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)
    parser.add_argument('--stackcount', dest='stack_count', type=int,
                        help='The stack number', required=True)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='Image type: either image or segmentation', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    image_num = args.image_num
    stackheight = args.stack_height
    stackby = args.stack_by
    varsinstack = args.vars_instack
    valsinstack = args.vals_instack
    imagepattern = args.image_pattern
    stackcount = args.stack_count
    imagetype = args.image_type
    # fpobject = args.fp_object

    # Initialize the logger    
    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(valsinstack),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO) 

    logger.info("Starting to build...")
    logger.info("{} values in stack are {}, respectively".format(varsinstack, valsinstack))


    # logger.info("Combos {}".format(com))
    fpobject = fp(input_dir, imagepattern, var_order=varsinstack)
    channels = []
    channelvals = []
    
    i = 0
    for item in fp.iterate(fpobject, group_by=stackby):
        if i == stackcount:
            for file in item:
                channels.append(Path(file['file']))
                base = os.path.basename(channels[-1])
                parsed = filepattern.parse_filename(base, pattern=imagepattern)
                channelvals.append(parsed[stackby])
            break
        else:
            i = i +1
    image = channels[0]

    logger.info("{} values in stack: {}".format(stackby, channelvals))

    # Make the output directory
    if pyramid_type == "Neuroglancer":
        out_dir = Path(output_dir).joinpath(image.name)
    elif pyramid_type == "DeepZoom":
        out_dir = Path(output_dir).joinpath('{}_files'.format(image_num))
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())

    # Create the BioReader object
    logger.info('Getting the BioReader...')
    bf = BioReader(str(image.absolute()),max_workers=max([cpu_count()-1,2]))
    depth = bf.num_z()

    # Create the output path and info file
    if pyramid_type == "Neuroglancer":
        file_info = utils.neuroglancer_info_file(bf,out_dir,stackheight, imagetype)
    elif pyramid_type == "DeepZoom":
        file_info = utils.dzi_file(bf,out_dir,image_num)
    else:
        ValueError("pyramid_type must be Neuroglancer or DeepZoom")
    logger.info("data_type: {}".format(file_info['data_type']))
    logger.info("num_channels: {}".format(file_info['num_channels']))
    logger.info("number of scales: {}".format(len(file_info['scales'])))
    logger.info("type: {}".format(file_info['type']))


    # Create the classes needed to generate a precomputed slice
    logger.info("Creating encoder and file writer...")
    if pyramid_type == "Neuroglancer":
        encoder = utils.NeuroglancerChunkEncoder(file_info)
        file_writer = utils.NeuroglancerWriter(out_dir)
        
    elif pyramid_type == "DeepZoom":
        encoder = utils.DeepZoomChunkEncoder(file_info)
        file_writer = utils.DeepZoomWriter(out_dir)

    # if imageType is segmentation, then populate ids with the label value
    ids = []
    # Create the stacked images
    if pyramid_type == "Neuroglancer":
        logger.info("Stack contains {} Levels (Stack's height)".format(stackheight))
        for i in range(0, stackheight):
            channelvalue = channelvals[i]
            channel = channels[i]

            if i == 0:
                utils._get_higher_res(0, channelvalue-1, bf, file_writer,encoder, imageType = imagetype, ids=ids, slices=[0,1])
            else:
                bf = BioReader(str(channels[i].absolute()))
                depth = bf.num_z()
                utils._get_higher_res(0, channelvalue-1, bf, file_writer,encoder, imageType = imagetype, ids=ids, slices=[0,1])

            if depth == 1:
                logger.info("Finished Level {}/{} in Stack ({})".format(i+1, stackheight, channel))
            else:
                logger.info("Finished Level {}/{} in Stack ({})".format(i+1, stackheight, channel))
                logger.info("IMAGE IN STACK HAD A Z DIMENSION OF {} UNITS".format(depth))
                logger.info("Only processed Z dimension [0, 1]")

        if imagetype == "segmentation":
            out_seginfo = utils.segmentinfo(encoder,ids,out_dir)

    logger.info("Finished precomputing. Closing the javabridge and exiting...")