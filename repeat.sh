rm -rf ./dataset/*
rm -rf ./relabledDataset/*
python3 "convert_cfg.py"

rm result.txt

for value in {1..4}
do
	python3 relabel_kmeans.py
	python3 src/graph2vec.py --input-path relabledDataset/ --output-path features/relabel.csv --dimensions 40 --epochs 500 --learning-rate 0.025
	for v2 in {1..5}
	do
		python3 -c 'from classify_neural_network import getaccuracy; getaccuracy() '
	done
done
python3 getstats.py
