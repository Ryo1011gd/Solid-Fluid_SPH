#!/bin/bash
set -e
../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4
