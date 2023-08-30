# -*- coding: utf-8 -*-
# This file is distributed under the MIT license.
#
# Created at Oct 10, 2022
# Author: Yuanfeng Zhang (zhangyuanfeng1997@foxmail.com) 
#
# Be careful: As this scirpt was based on polars v0.13.55, some functions and parameters had been replaced
# or removed in later versions. Thus, instead of running this script, we recommend writing a new script with
# the latest Python/Rust and polars library, following the instructions in Supplementary Methods.
#
# Usage: This script is used for cleansing the TCGA MC3 dataset.


import polars as pl
import re
import gc

# Read MC3 dataset, annotate every sample with tumor type, and only keep the following tumor types.
raw_mc3 = pl.read_csv('//your_path/mc3.v0.2.8.maf', sep='\t', infer_schema_length=0) \
    .join(other=pl.read_csv('//your_path/combined_study_clinical_data.tsv', sep='\t',
                            columns=['Tumor_Sample_Barcode', 'Study_Abbreviation'], infer_schema_length=0),
          left_on='Tumor_Sample_Barcode',
          right_on='Tumor_Sample_Barcode',
          how='left').filter(pl.col('Study_Abbreviation').is_in(['BLCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'SKCM',
                                                                 'STAD', 'UCEC', 'ACC', 'BRCA', 'CESC', 'ESCA',
                                                                 'GBM', 'KIRC', 'LGG', 'LIHC', 'PRAD', 'PEAD',
                                                                 'SARC', 'DLBC', 'OV', 'UCS']
                                                                )
                             ).with_columns(pl.col(['Start_Position', 'End_Position',
                                                    't_depth', 't_ref_count', 't_alt_count',
                                                    'n_depth', 'n_ref_count', 'n_alt_count']).cast(pl.Int32))

# Delete samples with 'native_wga_mix', 'wga', 'nonpreferredpair', 'gapfiller' tags.
mc3 = raw_mc3.filter(~pl.col('Tumor_Sample_Barcode')
                     .is_in(raw_mc3.filter(pl.col('FILTER')
                                           .str.contains("native_wga_mix|wga|nonpreferredpair|gapfiller")
                                           )['Tumor_Sample_Barcode'].unique())) \
    .with_column((pl.col('t_alt_count') / pl.col('t_depth')).alias('VAF'))

del raw_mc3
gc.collect()

# recollect samples, drop those the MSAF of which are below 10%.
enough_msaf_samples = mc3.groupby('Tumor_Sample_Barcode') \
    .agg(pl.col('VAF').sort(reverse=True).head(n=1).sum().suffix('_MSAF')) \
    .filter(pl.col('VAF_MSAF') > 0.1)['Tumor_Sample_Barcode']

enough_msaf_mc3 = mc3.filter(pl.col('Tumor_Sample_Barcode').is_in(enough_msaf_samples))

del mc3, enough_msaf_samples
gc.collect()

# recollect samples, drop those if half of their variants are labeled with 'StrandBias', 'common_in_exac', 'oxog'.
# And drop only the labeled variants if less than a half.
artifacts_count = enough_msaf_mc3.with_column(pl.lit(1).alias('temp_count')) \
    .groupby('Tumor_Sample_Barcode') \
    .pivot(pivot_column='FILTER', values_column='temp_count') \
    .sum()

artifacts_count = artifacts_count.fill_null(0).with_column(
    (pl.sum([col for col in artifacts_count.columns if re.search('StrandBias|common_in_exac|oxog', col)])
     /
     pl.sum([col for col in artifacts_count.columns if col != 'Tumor_Sample_Barcode'])
     )
    .alias('failed_proportion'))

labeled_mc3 = (enough_msaf_mc3
.filter(
    (pl.col('Tumor_Sample_Barcode')
     .is_in(artifacts_count.filter(pl.col('failed_proportion') < 0.5)['Tumor_Sample_Barcode']))
    &
    (~pl.col('FILTER')
     .str.contains("StrandBias|common_in_exac|oxog"))))

del enough_msaf_mc3, artifacts_count
gc.collect()

# Get rid of variants at extraneous sites. (but do not delete variants with low allele frequency)
qced_mc3 = (labeled_mc3
            .filter((pl.col('Variant_Classification').is_in(['Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del',
                                                             'In_Frame_Ins', 'Missense_Mutation', 'Nonsense_Mutation',
                                                             'Nonstop_Mutation', 'Splice_Site',
                                                             'Translation_Start_Site', 'Silent'])
                     )
                    &
                    (pl.col('t_depth') >= 25)
                    &
                    (pl.col('t_alt_count') >= 3)
                    # &
                    # ((pl.col('t_alt_count') / pl.col('t_depth')) >= 0.05)
                    )
            .with_column(pl.col('Tumor_Sample_Barcode').str.slice(0, 15).alias('array'))
            )

# Drop blank cols.
# noinspection PyTypeChecker
non_blank_mc3 = qced_mc3[[col for col in qced_mc3.columns if len(qced_mc3[col].unique()) != 1]]

# Add ABSOLATE tumor purity info.
absolate_table = pl.read_csv('//your_path/TCGA_mastercalls.abs_tables_JSedit.fixed.txt',
                             sep='\t')[['array', 'purity']]
final_mc3 = non_blank_mc3.lazy().join(absolate_table.lazy(), on='array', how='left').collect().drop(columns=['array'])

del non_blank_mc3, absolate_table, qced_mc3, labeled_mc3
gc.collect()

# Since WES TMB is immutable, calculate it this time.
class_for_westmb = ['Missense_Mutation', 'Nonsense_Mutation', 'In_Frame_Ins',
                    'In_Frame_Del', 'Frame_Shift_Del', 'Frame_Shift_Ins']
mc3_mutcount = (final_mc3
                .filter((pl.col('VAF') >= 0.05) & (pl.col('Variant_Classification').is_in(class_for_westmb)))
                .groupby('Tumor_Sample_Barcode')
                .count()
                .with_column((pl.col('count') / 34).alias('wesTMB'))
                [['Tumor_Sample_Barcode', 'wesTMB']]
                .lazy()
                )
final_mc3 = final_mc3.lazy().join(mc3_mutcount, how='left',
                                  left_on='Tumor_Sample_Barcode',
                                  right_on='Tumor_Sample_Barcode').collect()

del mc3_mutcount
gc.collect()

# rate TMB-H or TMB-L as True or False.
final_mc3 = final_mc3.with_column(pl.when((pl.col('wesTMB') < 10.0))
                                  .then(pl.lit(False))
                                  .otherwise(pl.lit(True))
                                  .alias('rate')
                                  )

# Sort mc3 file by wesTMB of samples.
mc3_sample = final_mc3.unique(subset='Tumor_Sample_Barcode').sort('wesTMB')['Tumor_Sample_Barcode'].to_list()
reordered_mc3 = pl.concat([final_mc3.filter(pl.col('Tumor_Sample_Barcode') == i) for i in mc3_sample])

del mc3_sample, final_mc3
gc.collect()

# Hash rows.
hashed_mc3 = (reordered_mc3
              .with_column(reordered_mc3.select(['Chromosome', 'Start_Position', 'Reference_Allele',
                                                 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode'])
                           .hash_rows()
                           .alias('hash')
                           )
              )
del reordered_mc3
gc.collect()

# write as tsv for general usage.
hashed_mc3.write_parquet('//your_path/hashed_mc3.parquet')

# generate file to retrieve snp database record.
hashed_mc3[['Chromosome',
            'Start_Position',
            'End_Position',
            'Reference_Allele',
            'Tumor_Seq_Allele2',
            'hash']].write_csv('//your_path/mc3_for_annovar.avinput',
                               sep='\t', has_header=False)
"""
use these cmdlns to annotate variants.
!!!
use annovar to annotate this file. (WSL Ubuntu 22.04 LTS)
cd ~/software/annovar

perl table_annovar.pl //your_path/mc3_for_annovar.avinput humandb/ \
  -out mc3_anno \
  -protocol ALL.sites.2015_08,AFR.sites.2015_08,AMR.sites.2015_08,EAS.sites.2015_08,EUR.sites.2015_08,SAS.sites.2015_08,exac03,gnomad211_exome,cosmic96_coding \
  --operation f,f,f,f,f,f,f,f,f\
  --nastring . --polish --thread 12 --otherinfo --remove -buildver hg19

mv mc3_anno.hg19_multianno.txt //your_path/mc3_anno.hg19_multianno.txt

!!!
"""

# merge Annovar results.
# noinspection PyTypeChecker
joined_mc3 = (hashed_mc3
              .with_column(pl.col('hash')
                           .cast(pl.Utf8))
              .lazy()
              .join(pl.read_csv('//your_path/mc3_for_annovar.avinput',
                                sep='\t', dtypes={'hash': pl.Utf8})[['ALL.sites.2015_08', 'AFR.sites.2015_08',
                                                                     'AMR.sites.2015_08', 'EAS.sites.2015_08',
                                                                     'EUR.sites.2015_08', 'SAS.sites.2015_08',
                                                                     'ExAC_ALL', 'ExAC_AFR', 'ExAC_AMR', 'ExAC_EAS',
                                                                     'ExAC_FIN', 'ExAC_NFE', 'ExAC_OTH', 'ExAC_SAS',
                                                                     'AF', 'AF_popmax', 'AF_male', 'AF_female',
                                                                     'AF_raw', 'AF_afr', 'AF_sas', 'AF_amr', 'AF_eas',
                                                                     'AF_nfe', 'AF_fin', 'AF_asj', 'AF_oth',
                                                                     'non_topmed_AF_popmax', 'non_neuro_AF_popmax',
                                                                     'non_cancer_AF_popmax', 'controls_AF_popmax',
                                                                     'cosmic96_coding', 'hash']].lazy(),
                    on='hash',
                    how='left')
              .collect()
              [['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Classification', 'Variant_Type',
                'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'dbSNP_RS', 'Tumor_Sample_Barcode',
                'Matched_Norm_Sample_Barcode', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2', 'HGVSc', 'HGVSp',
                'HGVSp_Short', 'Transcript_ID', 'Exon_Number', 't_depth', 't_ref_count', 't_alt_count', 'n_depth',
                'n_ref_count', 'n_alt_count', 'all_effects', 'Allele', 'Gene', 'Feature', 'Consequence',
                'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation',
                'STRAND', 'SYMBOL', 'SYMBOL_SOURCE', 'HGNC_ID', 'BIOTYPE', 'CANONICAL', 'CCDS', 'ENSP', 'SWISSPROT',
                'TREMBL', 'UNIPARC', 'SIFT', 'PolyPhen', 'EXON', 'INTRON', 'DOMAINS', 'ALL.sites.2015_08',
                'AFR.sites.2015_08', 'AMR.sites.2015_08', 'EAS.sites.2015_08', 'EUR.sites.2015_08', 'SAS.sites.2015_08',
                'AA_MAF', 'EA_MAF', 'ExAC_ALL', 'ExAC_AFR', 'ExAC_AMR', 'ExAC_EAS', 'ExAC_FIN', 'ExAC_NFE', 'ExAC_OTH',
                'ExAC_SAS', 'AF', 'AF_popmax', 'AF_male', 'AF_female', 'AF_raw', 'AF_afr', 'AF_sas', 'AF_amr', 'AF_eas',
                'AF_nfe', 'AF_fin', 'AF_asj', 'AF_oth', 'non_topmed_AF_popmax', 'non_neuro_AF_popmax',
                'non_cancer_AF_popmax', 'controls_AF_popmax', 'cosmic96_coding', 'CLIN_SIG', 'SOMATIC', 'PUBMED',
                'IMPACT', 'VARIANT_CLASS', 'HGVS_OFFSET', 'PHENO', 'GENE_PHENO', 'COSMIC', 'CENTERS', 'CONTEXT', 'DBVS',
                'NCALLERS', 'Study_Abbreviation', 'VAF', 'purity', 'wesTMB', 'rate', 'hash']]
              .rename({'ALL.sites.2015_08': 'GMAF', 'AFR.sites.2015_08': 'AFR_MAF', 'AMR.sites.2015_08': 'AMR_MAF',
                       'EAS.sites.2015_08': 'EAS_MAF', 'EUR.sites.2015_08': 'EUR_MAF', 'SAS.sites.2015_08': 'SAS_MAF',
                       'AF': 'gnomad', 'AF_popmax': 'gnomad_popmax', 'AF_male': 'gnomad_male',
                       'AF_female': 'gnomad_female', 'AF_raw': 'gnomad_raw', 'AF_afr': 'gnomad_afr',
                       'AF_sas': 'gnomad_sas', 'AF_amr': 'gnomad_amr', 'AF_eas': 'gnomad_eas',
                       'AF_nfe': 'gnomad_nfe', 'AF_fin': 'gnomad_fin', 'AF_asj': 'gnomad_asj',
                       'AF_oth': 'gnomad_oth'})
              .with_column(pl.when(pl.col('Consequence') == 'synonymous_variant')
                             .then(pl.lit('Synonymous_Mutation'))
                             .otherwise(pl.col('Variant_Classification'))
                             .alias('to_add')
                           )
              .drop('Variant_Classification')
              .rename({'to_add': 'Variant_Classification'}))

joined_mc3 = (joined_mc3[['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Classification']
                         +
                         [col for col in joined_mc3.columns if not col in ['Hugo_Symbol', 'Chromosome',
                                                                           'Start_Position', 'End_Position',
                                                                           'Variant_Classification']]])

del hashed_mc3
gc.collect()

# write it as parquet for data analysis. (Drop more cols to accelerate computing)
concise_cols = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Classification', 'Variant_Type',
                'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode',
                'Matched_Norm_Sample_Barcode', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2', 'HGVSc', 'HGVSp',
                'HGVSp_Short', 'Transcript_ID', 'Exon_Number', 't_depth', 't_ref_count', 't_alt_count', 'n_depth',
                'n_ref_count', 'n_alt_count', 'Allele', 'Consequence', 'cDNA_position', 'CDS_position',
                'Protein_position', 'STRAND', 'CCDS', 'EXON', 'INTRON', 'GMAF', 'AFR_MAF', 'AMR_MAF', 'EAS_MAF',
                'EUR_MAF', 'SAS_MAF', 'AA_MAF', 'EA_MAF', 'ExAC_ALL', 'ExAC_AFR', 'ExAC_AMR', 'ExAC_EAS', 'ExAC_FIN',
                'ExAC_NFE', 'ExAC_OTH', 'ExAC_SAS', 'gnomad', 'gnomad_popmax', 'gnomad_male', 'gnomad_female',
                'gnomad_raw', 'gnomad_afr', 'gnomad_sas', 'gnomad_amr', 'gnomad_eas', 'gnomad_nfe', 'gnomad_fin',
                'gnomad_asj', 'gnomad_oth', 'non_topmed_AF_popmax', 'non_neuro_AF_popmax', 'non_cancer_AF_popmax',
                'controls_AF_popmax', 'cosmic96_coding', 'CLIN_SIG', 'VARIANT_CLASS', 'CONTEXT',
                'Study_Abbreviation', 'purity', 'VAF', 'wesTMB', 'rate', 'hash']

minimal_cols = ['Chromosome', 'Start_Position', 'End_Position', 'Consequence', 'Variant_Classification',
                'Tumor_Sample_Barcode', 'purity', 'VAF', 'wesTMB', 'rate', 'hash']

joined_mc3.write_parquet('//your_path/complete_mc3.parquet', compression='lz4')
joined_mc3[concise_cols].write_parquet('//your_path/final_mc3.parquet', compression='lz4')
joined_mc3[minimal_cols].write_parquet('//your_path/minimum_mc3.parquet', compression='lz4')
