#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-3d-precomputed-labeling-plugin:${version}
