import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import utils
import filepattern
from filepattern import FilePattern as fp
import os
import itertools
import numpy as np
# import neuroglancer

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
    # parser.add_argument('--stackby', dest='stack_by', type=str,
    #                     help='Variable that the images get stacked by', required=False)
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)
    parser.add_argument('--image', dest='image', type=str,
                        help='The image to turn into a pyramid', required=True)
    parser.add_argument('--imagetype', dest='image_type', type=str,
                        help='The type of image: image or segmentation', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    image_num = args.image_num 
    # stackby = args.stack_by
    imagepattern = args.image_pattern
    image = args.image
    imagetype = args.image_type

    try:
        # Initialize the logger    
        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image_num),
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("build_pyramid")
        logger.setLevel(logging.INFO) 

        logger.info("Starting to build...")
        # logger.info("Values of xyz in stack are {}".format(valsinstack))

        # Initialize the javabridge
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

        # Create the BioReader object
        logger.info('Getting the BioReader...')
        bf = BioReader(str(Path(input_dir).joinpath(image)))
        imageread = bf.read_image()
        getimageshape = imageread.shape
        stackheight = getimageshape[2]

        logger.info(getimageshape)
        # mesh = np.dstack(bf.read_image())
        mesh = np.transpose(imageread, (0,2,1,3,4))
        # a = 0
        # for z in mesh:
        #     b = 0
        #     for y in z:
        #         c = 0
        #         for x in y:
        #             if x[0][0] != 0:
        #                 print(a, b, c, x[0][0])
        #             c = c + 1
        #         b = b + 1
        #     a = a + 1
        # for i in np.unique(mesh[:]):
            # print(i, np.unique(mesh[i]))
        ids = [int(i) for i in np.unique(mesh[:])]
        # print(ids)

        

        # Make the output directory
        if pyramid_type == "Neuroglancer":
            out_dir = Path(output_dir).joinpath(image)
            # out_dir = Path(output_dir)
        elif pyramid_type == "DeepZoom":
            out_dir = Path(output_dir).joinpath('{}_files'.format(image_num))
        out_dir.mkdir()
        out_dir = str(out_dir.absolute())

        # Create the output path and info file
        if pyramid_type == "Neuroglancer":
            file_info = utils.neuroglancer_info_file(bf,out_dir,stackheight, imagetype)
        elif pyramid_type == "DeepZoom":
            file_info = utils.dzi_file(bf,out_dir,image_num)
        else:
            ValueError("pyramid_type must be Neuroglancer or DeepZoom")

        numscales = len(file_info['scales'])
        logger.info("data_type: {}".format(file_info['data_type']))
        logger.info("num_channels: {}".format(file_info['num_channels']))
        logger.info("number of scales: {}".format(numscales))
        logger.info("type: {}".format(file_info['type']))
        
        
        # Create the classes needed to generate a precomputed slice
        logger.info("Creating encoder and file writer...")
        if pyramid_type == "Neuroglancer":
            encoder = utils.NeuroglancerChunkEncoder(file_info)
            file_writer = utils.NeuroglancerWriter(out_dir)
            out_seginfo = utils.segmentinfo(encoder,ids,out_dir)
        elif pyramid_type == "DeepZoom":
            encoder = utils.DeepZoomChunkEncoder(file_info)
            file_writer = utils.DeepZoomWriter(out_dir)
        
        # Create the stacked images
        if pyramid_type == "Neuroglancer":
            # out_seginfo = utils.segmentinfo(encoder,ids,out_dir)
            utils._get_higher_res(0, bf,file_writer,encoder)
            
        
        logger.info("Finished precomputing. Closing the javabridge and exiting...")
        jutil.kill_vm()
    except Exception as e:
        jutil.kill_vm()
        raise