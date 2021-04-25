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


# i2b2 2006

i2b2_2006_dir=../corpus/ner/2006_deidentification
i2b2_2006_train_dev=${i2b2_2006_dir}/deid_surrogate_train_all_version2.xml
i2b2_2006_train=${i2b2_2006_dir}/deid_surrogate_train_version2.xml
i2b2_2006_dev=${i2b2_2006_dir}/deid_surrogate_develop_version2.xml
i2b2_2006_test_original=${i2b2_2006_dir}/deid_surrogate_test_all_groundtruth_version2.xml
i2b2_2006_test=${i2b2_2006_dir}/deid_surrogate_test_version2.xml

python modules/dataset/xml_train_test_split.py \
    ${i2b2_2006_train_dev} \
    ${i2b2_2006_train} \
    ${i2b2_2006_dev} \
    --document-tag="record" 2>&1 | tee -a ${log_save_path}

python modules/dataset/preprocess_i2b2_2006_deidentification.py \
    ${i2b2_2006_train} \
    --overwrite 2>&1 | tee -a ${log_save_path}

python modules/dataset/preprocess_i2b2_2006_deidentification.py \
    ${i2b2_2006_dev} \
    --overwrite 2>&1 | tee -a ${log_save_path}

python modules/dataset/preprocess_i2b2_2006_deidentification.py \
    ${i2b2_2006_test_original} \
    ${i2b2_2006_test} \
    2>&1 | tee -a ${log_save_path}

python modules/dataset/convert_xml_to_samples.py \
    ${i2b2_2006_train} ${i2b2_2006_dir}/train \
    --document-tag="record" \
    2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py \
    ${i2b2_2006_dev} ${i2b2_2006_dir}/develop \
    --document-tag="record" \
    2>&1 | tee -a ${log_save_path}
python modules/dataset/convert_xml_to_samples.py \
    ${i2b2_2006_test} ${i2b2_2006_dir}/test \
    --document-tag="record" \
    2>&1 | tee -a ${log_save_path}
