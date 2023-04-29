data=$1 #EUR-Lex AmazonCat-13K Amazon-670K Wiki-500K


# Download axml data first
if [ $data == 'EUR-Lex' ]
then
    openid=1SriPq4G0xauFrvLLdiYPbLkxyueqnjFR
    data_pecos=eurlex-4k
elif [ $data == 'Wiki10-31K' ]
then
    openid=1aEMLP4tKt_EHLXJ1bYbG6CC_zBmptmDA
    data_pecos=wiki10-31k
elif [ $data == 'AmazonCat-13K' ]
then
    openid=185hCzc9Je96MOGB4JIAczuOWjC953LpR
    data_pecos=amazoncat-13k
elif [ $data == 'Wiki-500K' ]
then
    openid=1BkhZ08Vms4QDG2qkySmdZqrbmQ9Kq4q6
    data_pecos=wiki-500k
fi

data_path=$HOME/data/xml/data
mkdir -p $data_path

cd $data_path
echo $data_path

# Download xlgen data
gdown https://drive.google.com/uc?id=$openid&confirm=t

# Download Pecos data for evaluation
wget https://archive.org/download/pecos-dataset/xmc-base/${data_pecos}.tar.gz

# Unzip tar data
tar -zxvf ${data_path}/${data}.tar.gz
tar -zxvf ${data_path}/${data_pecos}.tar.gz

# Data folder move
mv ${data_path}/xmc-base/${data_pecos} ${data_path}/${data}
mv ${data_path}/${data}/${data_pecos} ${data_path}/${data}/pecos

#Remove unnecessary files
rm -rf ${data_path}/xmc-base
rm -rf ${data_path}/${data_pacos}.tar.gz
rm -rf ${data_path}/${data}.tar.gz

