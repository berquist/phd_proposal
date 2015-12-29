#!/usr/bin/env bash

# for f in $(ls *.cml | sort); do
#     echo ${f}
#     obabel ${f} -O ${f}.xyz
# done

for f in $(ls *.cml | sort); do
    echo ${f}
    obabel ${f} -O ${f}.inchi
    obabel ${f} -O ${f}.smiles
done
