#!/bin/bash
#for i in {1..10}
#do
	rm -rf and/*.png
	#rm -rf and_source/*.png
	rm -rf and_train/*.png
	rm -rf cross_train/*.png
	rm -rf cross/*.png
	#rm -rf cross_source/*.png
	rm -rf parallel-anti/*.png
	#rm -rf parallel-anti_source/*.png
	rm -rf parallel-anti_train/*.png
	rm -rf others/*.png
	#rm -rf others_source/*.png
	rm -rf others_train/*.png
	rm -rf random_true_test/*.png
	rm -rf random_true_train/*.png
	#echo $i >> info.txt
	#echo "---------" >> info.txt
	python produce.py
	#echo "produce is ok" >> info.txt
	python rotation.py
	#echo "rotation is ok" >> info.txt
	python merge.py
	#echo "merge is ok" >> info.txt
	python Random.py
	#echo "random is ok" >> info.txt
	#cd ..
	#python detect.py >>  info.txt
	#echo "Test" >> classify/info.txt
	#cd classify
#done

