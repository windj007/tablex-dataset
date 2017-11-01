#!/bin/bash

YOUR_HASHED_PASSWORD=sha1:43bb79968cf9:e8fb443c6f57e7c9f29ab112d8d4e0d5b5571fbf

BIND_PORT="-p 8890:8888"
if (( $# >= 1 ))
then
    BIND_PORT=""
fi


CMD="docker"
if which nvidia-docker
then
    CMD="nvidia-docker"
fi

$CMD run -ti --rm \
    -e "HASHED_PASSWORD=$YOUR_HASHED_PASSWORD" \
    -e "SSL=1" \
    -v `pwd`/certs:/jupyter/certs \
    -v `pwd`:/notebook \
    $BIND_PORT \
    -m 18G \
    --name tablex \
    --memory-swappiness 0 \
    --shm-size=8192m \
    tablex-dataset \
    $@
