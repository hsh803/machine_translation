import sacrebleu
import sys

def readfile(path):
    f = open(path,"r")
    text = f.readlines()
    f.close()
    return text

generated_file=sys.argv[1]
reference_file="/ateso-project/references.txt"
generated = readfile(generated_file)
reference = readfile(reference_file)

bleu = sacrebleu.corpus_bleu(generated, reference)

output_path = sys.argv[2]
output_file = open(output_path,"w")
output_file.write(f"BLEU score: {bleu.score:.2f}\n")
output_file.close()
print(f"BLEU score: {bleu.score:.2f}")
