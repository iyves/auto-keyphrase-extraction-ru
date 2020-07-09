#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 source_dir target_dir"
   exit 1 
}

if [ -z "$1" ] || [ -z "$2" ] || [ "$3" ]
then
   echo "Error: Missing or extraneous arguments";
   helpFunction
fi

for file in $1/*.pdf
do
   filename="$(basename ${file%.*})"
   pdftotext -y 80 -H 750 -W 1000 -nopgbrk -eol unix $file $2/$filename.txt
done
