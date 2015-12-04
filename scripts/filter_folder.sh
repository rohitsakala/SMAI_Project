#!/bin/bash

for i in ./full_phase_1/**/*; 
do
	tr -dc [:alnum:][\ ,.]\\n < $i >> temp
	mv temp $i
done
