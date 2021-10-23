#!/bin/bash
unzip -o -d case3 data.zip
mkdir -p data
mv case3/Aircraft_Classes_Private.csv data/AirCraftClasses_Public.csv
mv case3/Handling_Time_Private.csv data/Handling_Time_Public.csv
python3.8 /vagrant/aircraft_ms.py -c 3 -clm 30 -is 70 -rm 3 -sp 20 -st 4 $*
mv result-*.csv results.csv
grep 'total cost' stdout >> /dev/stderr
tar cjf logs.tbz *.txt *.csv stdout
