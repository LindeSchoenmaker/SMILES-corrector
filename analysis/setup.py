import os
import argparse

parser = argparse.ArgumentParser(description='Process SMILES')

parser.add_argument('-s',
                    '--source',
                    type=str,
                    default='generated/multi',
                    help='directory that contains file that want to check')
parser.add_argument('-d',
                    '--dest',
                    type=str,
                    default='generated/multi/error_analysis/',
                    help='where to save file with error messages')
parser.add_argument('-c',
                    '--column',
                    type=str,
                    default='INCORRECT',
                    help='column name of column containing the SMILES')
parser.add_argument('-f', 
                    '--files', 
                    nargs="*",
                    default=['transformer_all_1_PAPYRUS_200_16_3_gan_ckpt100_M_errors_200_S_fixed'],
                    help='names of files to check')
args = parser.parse_args()
setupcheck = 'python analysis/checksmiles.py'

source = f' -s "{args.source}"'
dest = f" 2> {args.dest}"
column = f' -c "{args.column}"'

files = args.files
for i, file in enumerate(files):
    src = f'{file}'
    dst = f'{file}.txt'
    call = setupcheck + source + ' -f ' + src + column + dest + dst
    print(call)
    os.system(call)
