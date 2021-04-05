log_save_path=create_dataset.log

python modules/dataset/brat_to_xml.py 2>&1 | tee -a ${log_save_path}

python modules/dataset/convert_xml_to_samples.py ../corpus/ncbi_disease_corpus/train.xml ../corpus/ncbi_disease_corpus/train 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/ncbi_disease_corpus/develop.xml ../corpus/ncbi_disease_corpus/develop 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/ncbi_disease_corpus/test.xml ../corpus/ncbi_disease_corpus/test 2>&1 | tee -a ${log_save_path}

python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/disease_train.xml ../corpus/bc5cdr/disease/train 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/disease_develop.xml ../corpus/bc5cdr/disease/develop 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/disease_test.xml ../corpus/bc5cdr/disease/test 2>&1 | tee -a ${log_save_path}


python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/chemical_train.xml ../corpus/bc5cdr/chemical/train 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/chemical_develop.xml ../corpus/bc5cdr/chemical/develop 2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py ../corpus/bc5cdr/chemical_test.xml ../corpus/bc5cdr/chemical/test 2>&1 | tee -a ${log_save_path}

