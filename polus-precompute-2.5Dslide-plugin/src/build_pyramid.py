import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import utils
import filepattern
from filepattern import FilePattern as fp
import os
import itertools



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
    # parser.add_argument('--varsinstack', dest='vars_instack', type=str,
    #                     help='Variables that the stack shares', required=True)
    parser.add_argument('--valinstack', dest='vals_instack', type=str, nargs='+',
                        help='Values of variables that the stack shares', required=True)
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)
    parser.add_argument('--stackcount', dest='stack_count', type=int,
                        help='The stack number', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    image_num = args.image_num
    stackheight = args.stack_height
    stackby = args.stack_by
    # varsinstack = args.vars_instack
    valsinstack = args.vals_instack
    imagepattern = args.image_pattern
    stackcount = args.stack_count
    # fpobject = args.fp_object


    # Initialize the logger    
    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(valsinstack),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO) 

    logger.info("Starting to build...")
    logger.info("Values of xyz in stack are {}".format(valsinstack))


    # logger.info("Combos {}".format(com))

    # Get images that are stacked together
    # filesbuild = filepattern.parse_directory(input_dir, pattern=imagepattern, var_order=varsinstack)
    # logger.info("Going into Recursive Function")
    # channels, channelvals, stackheight = utils.recursivefiles(filesbuild[0], varsinstack, valsinstack, stackby, stackheight, pattern=imagepattern)
    # image = channels[0]

    fpobject = fp(input_dir, imagepattern)
    channels = []
    channelvals = []
    i = 0
    for item in fp.iterate(fpobject, group_by='xyz'):
        if i == stackcount:
            for file in item:
                channels.append(Path(file['file']))
                base = os.path.basename(channels[-1])
                parsed = filepattern.parse_filename(base, pattern=imagepattern)
            break
        else:
            i = i +1
    image = channels[0]

    # allfiles = filepattern.parse_directory(input_dir, pattern=imagepattern, var_order=varsinstack+stackby)
    # all_varlists = [allfiles[1][item] for item in allfiles[1]]
    # all_combos = list(itertools.product(*all_varlists))

    # combos = {}
    # for x in all_combos:
    #     combos.setdefault(x[:len(varsinstack)], []).append(x)
    # combokeys = []
    # i = 0
    # potentialvalues = []
    # for item in combos:
    #     if i == stackcount:
    #         potentialvalues = combos[item]
    #         break
    #     else:
    #         continue
    # missingimages = set([vals[-1] for vals in potentialvalues]) - set(channelvals)

    # logger.info("CHECK: Potential {} values in stack: {}".format(stackby, [vals[-1] for vals in potentialvalues]))
    # for i in missingimages:
    #     logger.info("Potential Gaps in Stack: {}".format(i))
    for i in range(0, len(channels)):
        logger.info("Image {}: {}".format(i+1, channels[i]))
    

    # Initialize the javabridge
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    # Create the BioReader object
    logger.info('Getting the BioReader...')
    bf = BioReader(str(image.absolute()))

    # Create the output path and info file
    if pyramid_type == "Neuroglancer":
        file_info = utils.neuroglancer_info_file(bf,output_dir,stackheight)
    elif pyramid_type == "DeepZoom":
        file_info = utils.dzi_file(bf,output_dir,image_num)
    else:
        ValueError("pyramid_type must be Neuroglancer or DeepZoom")
    logger.info("data_type: {}".format(file_info['data_type']))
    logger.info("num_channels: {}".format(file_info['num_channels']))
    logger.info("number of scales: {}".format(len(file_info['scales'])))
    logger.info("type: {}".format(file_info['type']))

    # Make the output directory
    if pyramid_type == "Neuroglancer":
        # out_dir = Path(output_dir).joinpath(image.name)
        out_dir = Path(output_dir)
    elif pyramid_type == "DeepZoom":
        out_dir = Path(output_dir).joinpath('{}_files'.format(image_num))
    # out_dir.mkdir()
    out_dir = str(out_dir.absolute())
    
    
    # Create the classes needed to generate a precomputed slice
    logger.info("Creating encoder and file writer...")
    if pyramid_type == "Neuroglancer":
        encoder = utils.NeuroglancerChunkEncoder(file_info)
        file_writer = utils.NeuroglancerWriter(out_dir)
        logger.info(out_dir)
    elif pyramid_type == "DeepZoom":
        encoder = utils.DeepZoomChunkEncoder(file_info)
        file_writer = utils.DeepZoomWriter(out_dir)
    
    # Create the stacked images
    if pyramid_type == "Neuroglancer":
        logger.info("Stack contains {} Levels (Stack's height)".format(stackheight))
        for i in range(0, stackheight):
            if i == 0:
                utils._get_higher_res(0, i, bf,file_writer,encoder)
            else:
                bf = BioReader(str(channels[i].absolute()))
                # out_dir = Path(output_dir).joinpath(channels[i].absolute().name)
                # out_dir.mkdir()
                # out_dir = str(out_dir.absolute())
                # file_writer = utils.NeuroglancerWriter(out_dir)
                utils._get_higher_res(0, i, bf,file_writer,encoder)
            logger.info("Finished Level {} in Stack: {}".format(i, channels[i]))

    
    logger.info("Finished precomputing. Closing the javabridge and exiting...")
    jutil.kill_vm()