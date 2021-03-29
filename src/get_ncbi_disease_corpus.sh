repo_dir=`git rev-parse --show-toplevel`

# Download NCBI Disease Corpus

corpus_dir=${repo_dir}/corpus/ncbi_disease_corpus
mkdir -p ${corpus_dir}

train_zip_path=${corpus_dir}/NCBItrainset_corpus.zip
dev_zip_path=${corpus_dir}/NCBIdevelopset_corpus.zip
test_zip_path=${corpus_dir}/NCBItestset_corpus.zip

wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip -O ${train_zip_path}
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip -O ${dev_zip_path}
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip -O ${test_zip_path}


# Unzip NCBI Disease Corpus

unzip ${train_zip_path} -d ${corpus_dir} 
unzip ${dev_zip_path} -d ${corpus_dir}
unzip ${test_zip_path} -d ${corpus_dir}

rm ${train_zip_path}
rm ${dev_zip_path}
rm ${test_zip_path}
