import os
import os.path as osp
import pandas as pd
import subprocess
import pdb

root_dir = '/data/whx/projects/bert_prism/'
blast_dir = '/home/whx/Downloads/ncbi-blast-2.16.0+/'

predict_dir = root_dir + 'results/predicts/'
select_dir = root_dir + 'results/select/'

if not osp.exists(select_dir):
    os.makedirs(select_dir)

def sort_promoters():
    """
    Sorts promoter sequences based on authenticity and transcript level classifications.
    
    This function reads promoter sequence data and classification results from CSV files, combines them into a single DataFrame,
    and sorts them in descending order based on authenticity and transcript level classifications. The sorted results are then
    saved to a new CSV file.
    """
    sequences = pd.read_csv(osp.join(predict_dir, 'gen_seqs.csv'), delimiter=',', header=None)
    authenticity_cls = pd.read_csv(osp.join(predict_dir, 'authenticity_cls.csv'), delimiter=',', header=None)
    transcript_level_cls2 = pd.read_csv(osp.join(predict_dir, 'transcript_level_cls2.csv'), delimiter=',', header=None)
    transcript_level_cls4 = pd.read_csv(osp.join(predict_dir, 'transcript_level_cls4.csv'), delimiter=',', header=None)
    
    df = pd.concat([sequences, authenticity_cls, transcript_level_cls2, transcript_level_cls4], axis=1)
    df.columns = ['sequence', 'authenticity_cls', 'transcript_level_cls2', 'transcript_level_cls4']
    
    df_sorted = df.sort_values(by=['authenticity_cls', 'transcript_level_cls2', 'transcript_level_cls4'], ascending=[False, False, False])
    df_sorted.to_csv(osp.join(select_dir, 'predicts_sorted.csv'), sep=',', header=True)

def cutATG():
    """
    Reads data from 'predicts_sorted.csv', cuts each sequence at the start of a specified target sequence (ATG), 
    and saves the processed data to 'predicts_sorted_cut.csv'.
    
    The function identifies the position of the target sequence within each DNA sequence and retains only the portion 
    of the sequence before this target. The target sequence is defined by the variable `target_seq`.
    """
    file = pd.read_csv(osp.join(select_dir, 'predicts_sorted.csv'))
    sequences = file['sequence'].tolist()
    
    # atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa
    target_seq = 'ATGGTGAGCAAGGGCGAGGA'
    # target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGT'
    # target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGG'

    cutSequences = []
    for seq in sequences:
        position = seq.find(target_seq)
        promoter = seq[:position]
        cutSequences.append(promoter)
    
    file['sequence'] = cutSequences
    file.to_csv(os.path.join(select_dir, 'predicts_sorted_cut.csv'), index=False)

def csv2fasta():
    """
    Converts sequence data from a CSV file into FASTA format.

    This function reads sequence data from 'predicts_sorted_cut.csv' and writes it to a new file named 
    'predicts_sorted_cut.fasta' in FASTA format, using identifiers from the first column of the CSV as sequence headers.
    """
    file = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut.csv'))
    sequences = file['sequence'].tolist()
        
    f = open(osp.join(select_dir, 'predicts_sorted_cut.fasta'), 'w')
    # number = 1
    for i in range(len(sequences)):
        seq_cut = sequences[i].replace('\n', '')
        f.write(f'>{file.iloc[i,0]}\n')
        f.write(f'{seq_cut}\n')
        # number += 1
    f.close()

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {command}")

def blast():
    """
    Performs BLAST analysis of query sequences against a reference genome database.

    This function runs BLASTN to align sequences from 'predicts_sorted_cut.fasta' against a nucleotide database created 
    from 'GCF_000005845.2_ASM584v2_genomic.fna'. Results are saved in a tabular format to 'predicts_sorted_cut_blast.txt'.
    """
    fasta_path = osp.join(select_dir, 'predicts_sorted_cut.fasta')
    
    # Escherichia coli K-12 MG1655 T00007
    fna_path = osp.join(root_dir, '2select/fasta/GCF_000005845.2_ASM584v2_genomic.fna')

    run_command(f'{blast_dir}bin/makeblastdb -in {fna_path} -dbtype nucl')
    run_command(f'{blast_dir}bin/blastn -query {fasta_path} -db {fna_path} -out {select_dir}/predicts_sorted_cut_blast.txt -outfmt 6')

def blast_filter():
    """
    Filters BLAST results to retain only the first occurrence of each query sequence.

    This function reads BLAST output from 'predicts_sorted_cut_blast.txt', removes duplicate entries for the same query,
    and saves the filtered results to 'predicts_sorted_cut_blast.csv' with appropriate column headers.
    """
    path = osp.join(select_dir, 'predicts_sorted_cut_blast.txt')
    f = open(path)
    file = f.readlines()
    f.close()
    
    selected_list = []
    selected_lines = []
    for line in file:
        line_cut = line.replace('\n', '').split('\t')
        if not line_cut[0] in selected_list:
            selected_list.append(line_cut[0])
            selected_lines.append(line_cut)
        else:
            continue

    df = pd.DataFrame(selected_lines)
    df.columns = [
        'Query ID',
        'Subject ID',
        '% Identity',
        'Alignment Length',
        'Mismatches',
        'Gap Opens',
        'Query Start',
        'Query End',
        'Subject Start',
        'Subject End',
        'E-value',
        'Bit Score',
    ]
    df.to_csv(os.path.join(osp.join(select_dir, 'predicts_sorted_cut_blast.csv')), index=False, header=True)

def select_blast():
    """
    Merges BLAST results with the original sequence data based on query identifiers.

    This function combines data from 'predicts_sorted_cut.csv' and 'predicts_sorted_cut_blast.csv' using a left join on 
    query identifiers, and saves the merged dataset to 'predicts_sorted_cut_blast_merge.csv'.
    """
    df_blast = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut_blast.csv'))
    df_seqs = pd.read_csv(osp.join(select_dir, 'predicts_sorted_cut.csv'))
    
    df_merged = pd.merge(
        df_seqs,
        df_blast,
        left_on='Unnamed: 0',
        right_on='Query ID',
        how='left'
    )    
    df_merged.to_csv(osp.join(select_dir, 'predicts_sorted_cut_blast_merge.csv'), index=False)


if __name__ == '__main__':
    sort_promoters()
    cutATG()
    csv2fasta()
    blast()
    blast_filter()
    select_blast()
