set -e

dataset=../../winost_metadata/en_subset.tsv
lang=$1
trans_sys=$2

prefix=en-$lang
trans_fn=/livingrooms/ycevan/speech_llm_bias_eval_data/mt/translations/$trans_sys/$prefix.txt

# Align
align_fn=forward.$prefix.align
/home/ycevan/speech_llm_bias_eval/mt/mt_gender/fast_align/build/fast_align -i $trans_fn -d -o -v > $align_fn

# Evaluate
mkdir -p ../data/human/$trans_sys/$lang/
out_fn=../data/human/$trans_sys/$lang/${lang}.pred.csv
python ../src/load_alignments.py --ds=$dataset  --bi=$trans_fn --align=$align_fn --lang=$lang --out=$out_fn