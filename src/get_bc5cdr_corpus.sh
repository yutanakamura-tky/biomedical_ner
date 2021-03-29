repo_dir=`git rev-parse --show-toplevel`

# Download BC5CDR Corpus

corpus_dir=${repo_dir}/corpus/bc5cdr
mkdir -p ${corpus_dir}

zip_path=${corpus_dir}/CDR_Data.zip

wget http://www.biocreative.org/media/store/files/2016/CDR_Data.zip -O ${zip_path}


# Unzip BC5CDR Corpus

unzip ${zip_path} -d ${corpus_dir} 
rm ${zip_path}


# Remove Unneccessary Directory
rm -r ${corpus_dir}/__MACOSX


# Rearrange Directory
mv ${corpus_dir}/CDR_Data/* ${corpus_dir}
rm -r ${corpus_dir}/CDR_Data

mv ${corpus_dir}/CDR.Corpus.v010516/* ${corpus_dir}
rm -r ${corpus_dir}/CDR.Corpus.v010516

mv ${corpus_dir}/DNorm.TestSet/* ${corpus_dir}
rm -r ${corpus_dir}/DNorm.TestSet

mv ${corpus_dir}/tmChem.TestSet/* ${corpus_dir}
rm -r ${corpus_dir}/tmChem.TestSet
