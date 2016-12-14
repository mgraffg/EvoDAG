#!/bin/bash

cd $RECIPE_DIR
# echo "Building !!!!" `pwd`
# $PYTHON setup.py build_ext
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
