#!/bin/bash

function download-data() {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

download-data 1uJDqW4u2shjbA_Y27oIcq-QM2jgW4c9g electricity.npy
download-data 1R_M3RP-t5CKDj5gUVTU2IEZXDInQKHJG traffic.npy
