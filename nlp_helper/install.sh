python3 -m pip install -r requirements.txt
fileid="1swWtUjFAh66GuYWM-ZJMsFYlmvgE6U2j"
filename="roberta_few_labels.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" 
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip roberta_few_labels.zip
rm roberta_few_labels.zip
