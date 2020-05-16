import io
from os import listdir


line_index = 1
f = open("../testdata/few_labels/dev.txt", "w")
lst = ["O", "B-Diagnostic_procedure", "I-Diagnostic_procedure","B-Biological_structure", "I-Biological_structure", "B-Sign_symptom", "I-Sign_symptom", "B-Detailed_description", "I-Detailed_description", "B-Lab_value", "I-Lab_value", "B-Date", "I-Date", "B-Age", "I-Age", "B-Clinical_event", "I-Clinical_event", "B-Date", "I-Date", "B-Disease_disorder", "I-Disease_disorder", "B-Nonbiological_location", "I-Nonbiological_location", "B-Severity", "I-Severity", "B-Sex", "B-Therapeutic_procedure", "I-Therapeutic_procedure"]
with open("../testdata/dev.txt", "r") as readFile:
    for line in readFile:
        word = line.split()
        print(word)
        if len(word) == 0:
            continue
        try:
            if word[1] not in lst:
                word[1] = 'O'
            wordString = ' '.join(word) 
            f.writelines('%s\n' %wordString)
            """
            if line[1] == "O":
                writer.writerow(["Sentence: "+str(line_index), line[0], "", line[1]])
            elif line[1][2:] in lst2:
                writer.writerow(["Sentence: "+str(line_index), line[0], "", "O"])
            else:
                writer.writerow(["Sentence: "+str(line_index), line[0], "", line[1][2:]])
            """
            #writer.writerow(["Sentence: "+str(line_index), line[0], "", line[1]])
        except:
            continue



