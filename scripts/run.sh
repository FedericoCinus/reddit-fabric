python create_dataset.py --years "[2016,2017,2018,2019,2020]" --attribute "gender__body" --types_of_text "['submissions','comments']"
python train.py --use_bool_features true --test_size 0.2 --attribute_path "../data/dataset/gender__body"
python create_dataset.py --years "[2016,2017,2018,2019,2020]" --attribute "demo_rep__body" --types_of_text "['submissions','comments']"
python train.py --use_bool_features true --test_size 0.2 --attribute_path "../data/dataset/demo_rep__body"
python create_dataset.py --years "[2016,2017,2018,2019,2020]" --attribute "age__body" --types_of_text "['submissions','comments']"
python train.py --use_bool_features true --test_size 0.2 --attribute_path "../data/dataset/age__body" --use_bool_features False
python train_and_test_classifiers.py -s 100 -a