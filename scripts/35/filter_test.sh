#!/bin/bash

for i in ./full_phase_1_test/*; 
do
	tr -dc [:alnum:][\ ,.]\\n < $i >> temp
	mv temp $i
done
rm temp
