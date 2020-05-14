import csv
import io
from os import listdir


file = open("train_stanford-test.txt", "r", encoding='utf-8')
text = file.readlines()


line_index = 1
with open("train_stanford4-test.csv", "w+") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(["Sentence #", "Word", "POS", "Tag"])
    lst = ["O", "B-Diagnostic_procedure", "I-Diagnostic_procedure","B-Biological_structure", "I-Biological_structure", "B-Sign_symptom", "I-Sign_symptom", "B-Detailed_description", "I-Detailed_description", "B-Lab_value", "I-Lab_value", "B-Date", "I-Date", "B-Age", "I-Age", "B-Clinical_event", "I-Clinical_event", "B-Date", "I-Date", "B-Disease_disorder", "I-Disease_disorder", "B-Nonbiological_location", "I-Nonbiological_location", "B-Severity", "I-Severity", "B-Sex", "B-Therapeutic_procedure", "I-Therapeutic_procedure"]
    #lst2 = ["Volume", "Time", "Texture", "Shape", "Quantitative_concept", "Qualitative_concept", "Outcome", "Other_event", "Other_entity", "Coreference", "Color", "Biological_attribute", "Administration", "Activity"]
    for i in range(len(text)):
        line = text[i].split()
        if len(line) == 0:
            continue
        try:
            if line[1] not in lst:
                writer.writerow(["Sentence: "+str(line_index), line[0], "", "O"])
            else:
                writer.writerow(["Sentence: "+str(line_index), line[0], "", line[1]])
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
        if line[0] == '.' and len(text[i+1].split()) == 0:
            line_index+=1


