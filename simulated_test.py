# -*- coding: utf-8 -*-
from itertools import combinations, product, repeat
from multiprocessing import Pool
from pathlib import Path
from pickle import dump, load
from random import randint
from typing import Literal

from numpy import float64, maximum, mean, nan, std, var
from pandas import DataFrame, read_excel, read_parquet, read_table, Series
#  from pybedtools import BedTool, cleanup, set_tempdir
from pybedtools import BedTool, set_tempdir
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from scipy.stats import pearsonr
from tqdm import tqdm

set_tempdir('./pybedtools_tmp')

driver_gene_set = {'MET', 'SBDS', 'GRM3', 'BCL6', 'EPHA7', 'RANBP2', 'FBXO11', 'NUP98', 'FAM135B', 'BMPR1A', 'MTCP1',
                   'FAT3', 'CD74', 'PCBP1', 'PHF6', 'SGK1', 'RAD21', 'ERCC4', 'KEAP1', 'CASP3', 'PALB2', 'RARA', 'MPL',
                   'CREB3L2', 'CBFA2T3', 'FBXW7', 'PTPN6', 'CCND3', 'PBX1', 'PRDM1', 'CYLD', 'RGPD3', 'TRIM33', 'WT1',
                   'XPA', 'CHD4', 'ATR', 'SSX1', 'PPP6C', 'AFF4', 'ATRX', 'JAK1', 'ROBO2', 'MLLT10', 'PRPF40B', 'DEK',
                   'CDKN1A', 'SETDB1', 'MAPK1', 'PIK3CB', 'DDB2', 'NBEA', 'TFE3', 'SETD1B', 'BCL11A', 'QKI', 'PATZ1',
                   'ERCC3', 'CARS1', 'EPHA3', 'CTNNA2', 'FOXP1', 'PRDM2', 'GRIN2A', 'FAS', 'SMO', 'SET', 'NCOA4',
                   'LCK', 'FOXO4', 'CHEK2', 'ETNK1', 'NFE2L2', 'PTPRT', 'SMAD3', 'STK11', 'FOXR1', 'IKZF1', 'ALK',
                   'LYL1', 'FOXA1', 'RMI2', 'GNAQ', 'ARHGEF10', 'BAZ1A', 'ABI1', 'SIX1', 'ROS1', 'ELF3', 'PBRM1',
                   'ACVR1B', 'TRIM24', 'ZFHX3', 'ID3', 'PDGFRB', 'YWHAE', 'FOXO3', 'ITGAV', 'FCGR2B', 'ZRSR2', 'LATS2',
                   'RFWD3', 'HOXA13', 'PTPRD', 'TGFBR2', 'CAMTA1', 'CHD2', 'PTEN', 'TCL1A', 'CDH10', 'CDK6', 'TPM3',
                   'TEC', 'DAXX', 'EED', 'PML', 'NTHL1', 'VAV1', 'SMAD2', 'FANCE', 'RAD51B', 'MAX', 'TET1', 'CDK4',
                   'POU5F1', 'MAFB', 'RAF1', 'FANCG', 'FAM47C', 'FLT4', 'IRF4', 'AFF3', 'CHST11', 'BRAF', 'SDHD',
                   'CDC73', 'DDX10', 'ARAF', 'PIM1', 'PRF1', 'PMS1', 'EZH2', 'ARNT', 'TERT', 'CCNE1', 'AKT2', 'MGMT',
                   'FLI1', 'BMP5', 'CUX1', 'REL', 'RSPO3', 'DDX5', 'EPS15', 'A1CF', 'PRDM16', 'BTG1', 'CNBD1', 'WAS',
                   'CCNB1IP1', 'BCOR', 'SLC34A2', 'HMGA2', 'NTRK2', 'HRAS', 'NOTCH1', 'PRKAR1A', 'BCL11B', 'DDX3X',
                   'TLX1', 'ARHGEF12', 'WWTR1', 'FGFR4', 'NFKBIE', 'PIK3CA', 'LZTR1', 'CCR4', 'MN1', 'PRKCB', 'HOXD11',
                   'TMEM127', 'ZNF521', 'RGS7', 'TAF15', 'SIRPA', 'ATM', 'STAG1', 'TNFAIP3', 'TAL2', 'N4BP2', 'POLE',
                   'SMARCD1', 'CD209', 'FKBP9', 'PTPRC', 'CIITA', 'FUBP1', 'BIRC3', 'FANCA', 'SFPQ', 'TRRAP', 'SOCS1',
                   'DCAF12L2', 'CREB3L1', 'PTPRK', 'KIT', 'HNF1A', 'PTK6', 'SKI', 'MTOR', 'KLF6', 'CALR', 'TSHR',
                   'PAX5', 'AFDN', 'MACC1', 'NPM1', 'UBR5', 'ARHGAP26', 'DDX6', 'PHOX2B', 'SUZ12', 'CDKN1B', 'SMAD4',
                   'PPM1D', 'CEBPA', 'CSF1R', 'CTNND2', 'PPP2R1A', 'STAG2', 'KAT7', 'RPL5', 'HMGA1', 'CRTC1', 'EP300',
                   'CDH11', 'ERCC2', 'LMO2', 'NOTCH2', 'VHL', 'DNM2', 'MYB', 'NRAS', 'MEN1', 'HEY1', 'LATS1', 'CBLC',
                   'FH', 'ATF1', 'RAP1GDS1', 'MLH1', 'DICER1', 'CDX2', 'MSH2', 'AMER1', 'BIRC6', 'FGFR3', 'CDKN2C',
                   'CLTCL1', 'ELK4', 'ESR1', 'FANCC', 'MYOD1', 'RHOH', 'STIL', 'USP8', 'H3C2', 'H3-3B', 'FES', 'TRIM27',
                   'BRD4', 'SETBP1', 'CREBBP', 'HOXA11', 'TLX3', 'PIK3R1', 'KRAS', 'SETD2', 'SFRP4', 'PDGFRA', 'BCL10',
                   'BTK', 'POU2AF1', 'MYCL', 'APC', 'CREB1', 'HLF', 'MDM4', 'CCNC', 'P2RY8', 'RIT1', 'RSPO2', 'EXT1',
                   'LEF1', 'POLD1', 'ZNRF3', 'TFEB', 'NDRG1', 'CD79B', 'PLCG1', 'KAT6A', 'MAP2K1', 'MSH6', 'RUNX1T1',
                   'SMARCB1', 'TCF3', 'CD274', 'MAP2K4', 'RPL22', 'CTNND1', 'KDR', 'SF3B1', 'USP6', 'CTNNB1', 'CDH1',
                   'TET2', 'PTPRB', 'CARD11', 'PABPC1', 'PIK3R2', 'MRTFA', 'WNK2', 'TCF7L2', 'SOX21', 'H3-3A', 'CUL3',
                   'DROSHA', 'ELL', 'PDCD1LG2', 'PTPN11', 'SUFU', 'CASP9', 'OLIG2', 'NFATC2', 'APOBEC3B', 'NF1',
                   'PDGFB', 'ACVR1', 'HOXC11', 'CDK12', 'CDH17', 'SMARCA4', 'FLT3', 'MSI2', 'GPC3', 'SRSF2', 'FLNA',
                   'FANCF', 'PREX2', 'GLI1', 'SSX4', 'CXCR4', 'ANK1', 'LMO1', 'MED12', 'IL7R', 'STAT6', 'SPEN', 'HIP1',
                   'MAP3K13', 'CNOT3', 'DDR2', 'BRIP1', 'EWSR1', 'FUS', 'BCORL1', 'GNAS', 'NT5C2', 'DNMT3A', 'NCOA2',
                   'SIX2', 'SDHA', 'MYD88', 'LRP1B', 'TAL1', 'HIF1A', 'STAT5B', 'NBN', 'RNF43', 'FAT4', 'IDH2', 'DDIT3',
                   'KMT2C', 'ARID2', 'BCL2', 'EIF3E', 'CCND1', 'LEPROTL1', 'NSD3', 'SDHC', 'USP44', 'ABL2', 'AXIN2',
                   'CCDC6', 'BAX', 'ARHGAP5', 'ECT2L', 'CASP8', 'TP53', 'CCR7', 'PTCH1', 'SPOP', 'ARID1A', 'TNFRSF14',
                   'PSIP1', 'TSC2', 'GATA3', 'SYK', 'RECQL4', 'ASXL2', 'PPARG', 'NCOR2', 'SND1', 'XPC', 'TRAF7',
                   'CACNA1D', 'HNRNPA2B1', 'DCC', 'CLTC', 'AKT3', 'KMT2D', 'STAT3', 'RBM10', 'NCOR1', 'CYP2C8',
                   'RPL10', 'SRSF3', 'NTRK3', 'ACVR2A', 'IKBKB', 'CRNKL1', 'NTRK1', 'ABL1', 'ATP2B3', 'HOXC13',
                   'S100A7', 'POT1', 'SMC1A', 'TSC1', 'CDKN2A', 'U2AF1', 'SH2B3', 'ZNF331', 'RB1', 'ZNF479', 'BCL2L12',
                   'BRCA1', 'SRC', 'CRLF2', 'GATA2', 'AKT1', 'DNMT1', 'PRKACA', 'SSX2', 'TP63', 'TBX3', 'MAML2',
                   'AXIN1', 'DGCR8', 'FOXL2', 'EIF1AX', 'KDM6A', 'LPP', 'ETV1', 'TNC', 'NUTM1', 'KLF4', 'WRN', 'ETV5',
                   'EXT2', 'IGF2BP2', 'ERCC5', 'IL6ST', 'MALT1', 'BCL3', 'PER1', 'BLM', 'ETV4', 'ASXL1', 'CPEB3',
                   'MAP2K2', 'CIC', 'BRD3', 'BCL9L', 'FEN1', 'BARD1', 'CBFB', 'CTCF', 'FAT1', 'GATA1', 'COL2A1', 'ELF4',
                   'CCND2', 'ZEB1', 'FOXO1', 'PMS2', 'NSD2', 'AR', 'EBF1', 'KDM5A', 'MAF', 'ISX', 'CYSLTR2', 'CD79A',
                   'HOXD13', 'NAB2', 'SOX2', 'TBL1XR1', 'BUB1B', 'EGFR', 'NR4A3', 'PLAG1', 'KNSTRN', 'FGFR1', 'SALL4',
                   'TNFRSF17', 'ACKR3', 'RET', 'CSF3R', 'MECOM', 'NKX2-1', 'FLCN', 'MUTYH', 'MLF1', 'SMARCE1', 'FCRL4',
                   'NF2', 'ZBTB16', 'FEV', 'ERBB3', 'BCLAF1', 'KCNJ5', 'ARHGEF10L', 'FGFR2', 'ETV6', 'PAX3', 'NFKB2',
                   'PTPN13', 'SOX9', 'FSTL3', 'POLG', 'SDHB', 'MYH9', 'KMT2A', 'MDM2', 'JUN', 'SH3GL1', 'ZNF429',
                   'FANCD2', 'BRCA2', 'XPO1', 'JAK3', 'MYC', 'ZMYM3', 'BCL9', 'HOXA9', 'TENT5C', 'MYCN', 'ATP1A1',
                   'ERBB2', 'GNA11', 'BAP1', 'KAT6B', 'CBLB', 'ERG', 'ARID1B', 'MITF', 'LARP4B', 'POLQ', 'JAK2',
                   'KDM5C', 'SDHAF2', 'ERBB4', 'RUNX1', 'CD28', 'MAP3K1', 'NRG1', 'IDH1', 'FBLN2', 'EPAS1', 'WIF1',
                   'GPC5', 'B2M', 'FHIT', 'RAD17', 'CBL', 'KNL1', 'IRS4', 'LRIG3', 'RAC1', 'RHOA', 'CNBP'}


def get_bed_files(bed_files_dir: str, prefix_list: list[str]):
    _files: list[Path] = [file for file in list(Path(bed_files_dir).glob('panel*bed')) if file.stem[:7] in prefix_list]
    if not _files:
        raise FileNotFoundError(f'folder {bed_files_dir} is empty!')
    _files_paths: list[str] = [str(_file) for _file in _files]

    return _files_paths


def random_systematic_sampling(ls: list, the_len: int = 9):
    to_pick = []
    for basic in range(0, len(ls), the_len):
        seed = randint(0, the_len - 1)
        to_pick.append(basic + seed)
    to_pick.pop(-1)
    not_picked = set(range(0, len(ls))) - set(to_pick)
    return list(map(ls.__getitem__, not_picked)), list(map(ls.__getitem__, to_pick))


def parquet2bedtool(parquet_file: str) -> BedTool:
    _df: DataFrame = (read_parquet(tr_file).drop(['Consequence', 'hash'], axis=1).query('wesTMB <=100.0'))
    _df['Start_Position'] -= 1
    return BedTool.from_dataframe(_df)


class BedFilePreset():
    def __init__(self, bed_file_path: str):
        self.file_path: str = bed_file_path
        self._check_file_path()
        self.name: str = bed_file_path.split('/')[-1].split('_')[0]   # for the EQA only.
        self.df: DataFrame = self._read_as_df()
        self.bedtool: BedTool = BedTool.from_dataframe(self.df)
        self.bed_length: int = self.get_length()

    def _check_file_path(self):
        if not Path(self.file_path).exists():
            raise (FileNotFoundError(f'wrong bed file path, cannot construct it:\n{self.file_path}'))

    def _read_as_df(self):
        return read_table(self.file_path,
                          names=['chr', 'start', 'end'],
                          dtype={'#chrom': 'str'})

    @staticmethod
    def calculate_bed_length(bed_df: DataFrame,
                             start_col: str = 'Start',
                             end_col: str = 'End') -> int:
        bed_df['length'] = bed_df.copy().apply(lambda col: col[end_col] - col[start_col], axis=1)
        return bed_df['length'].sum() / 1e6

    def get_length(self):
        return self.calculate_bed_length(self.df, start_col='start', end_col='end')

    @staticmethod
    def intersect_with_bedtool(dataframe_obj: DataFrame, bedtool_obj: BedTool,
                               mode: Literal['error', 'warn'], **kwargs):
        if kwargs:
            _intersect_df: DataFrame = bedtool_obj.intersect(BedTool.from_dataframe(dataframe_obj),  # type: ignore
                                                             wa=True).to_dataframe(disable_auto_names=True)
        else:
            _intersect_df: DataFrame = bedtool_obj.intersect(BedTool.from_dataframe(dataframe_obj),  # type: ignore
                                                             wa=True).to_dataframe(disable_auto_names=True)
        if _intersect_df.empty:
            if mode == 'error':
                raise ValueError('Intersection is empty')
            elif mode == 'warn':
                print('Intersection is empty')
            else:
                pass
        else:
            return _intersect_df

    def get_dataset(self, dataset_bedtool: BedTool) -> DataFrame:
        _df: DataFrame = (dataset_bedtool
                          .intersect(self.bedtool, wa=True)  # type: ignore
                          .to_dataframe(names=['chr', 'start', 'end', 'type',
                                               'Tumor_Sample_Barcode', 'purity',
                                               'vaf', 'wesTMB', 'rate', 'max_maf',
                                               'cosmic96_coding', 'cancerhotspot', 'civic',
                                               'Hugo_Symbol'],
                                        dtype={'max_maf': 'str',
                                               'cosmic96_coding': 'str'}))
        _df['max_maf'] = _df['max_maf'].apply(lambda x: nan if x == '.' else float(x))
        _df['cosmic96_coding'] = _df['cosmic96_coding'].apply(lambda x: 0 if x == '.' else float(x))
        _df['purity'] = _df['purity'].apply(lambda x: nan if x == '.' else x)
        return _df

    @staticmethod
    def construct_params(somatic_accuracy: bool,
                         gene_content: bool,
                         hotspot_filter: bool,
                         vaf_cutoff: bool) -> list:
        # deleted: filter_rules: bool,
        _recall_list: list[float]
        _precision_list: list[float]
        if somatic_accuracy:
            _recall_list = []
            _precision_list = []
            _scale1: tuple = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
            # _scale2: tuple = (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91)
            # for _pair in (list(product(_scale1, repeat=2)) + list(product(_scale2, repeat=2))):
            for _pair in list(product(_scale1, repeat=2)):
                if _pair[0] > _pair[1] or _pair[0] == _pair[1] == 1.0:
                    _recall_list.append(_pair[0])
                    _precision_list.append(_pair[1])
        else:
            _recall_list = [1.0]
            _precision_list = [1.0]

        _basic_gene_content: list = ['Frame_Shift_Del', 'Frame_Shift_Ins',
                                     'In_Frame_Del', 'In_Frame_Ins', 'Missense_Mutation']

        _gene_content_list: list
        if gene_content:
            _gene_content_list = []
            _available_gene_content: set = {'Nonsense_Mutation', 'Nonstop_Mutation', 'Synonymous_Mutation',
                                            'Silent', 'Splice_Site', 'Translation_Start_Site'}
            for _f in range(1, 7):
                _gene_content_list += [_basic_gene_content + list(_i) for _i in combinations(_available_gene_content,
                                                                                             _f)]
        else:
            _gene_content_list = [_basic_gene_content]

        # _pmaf_list: list[float]
        # _driver_list: list[bool]
        # _hotspot_list: list[bool]
        # _cosmic_threshold_list: list[int]
        # if filter_rules:
        #     _pmaf_list = [0.01, 0.005, 0.0001, .0, -1.0]
        #     _driver_list = [True, False]
        #     _hotspot_list = [True, False]
        #     _cosmic_threshold_list = [10, 20, 50, 100]

        # else:
        #     _pmaf_list: list[float] = [-1.0]
        #     _driver_list = [False]
        #     _hotspot_list = [False]
        #     _cosmic_threshold_list = [0]
        _hotspot_list: list[bool]
        if hotspot_filter:
            _hotspot_list = [True, False]
        else:
            _hotspot_list = [False]

        _vaf_cutoff_list: list[float]
        if vaf_cutoff:
            _vaf_cutoff_list = [0.01, 0.02, 0.03, 0.05, 0.10]
        else:
            _vaf_cutoff_list = [0.05]

        _params: list = list(product(_recall_list, _precision_list, _gene_content_list,
                                     #  _pmaf_list, _driver_list, _hotspot_list, _cosmic_threshold_list,
                                     _hotspot_list, _vaf_cutoff_list))
        return _params

    @staticmethod
    def calculate_ccc(y_true: Series, y_pred: Series) -> float:
        return float(2 * pearsonr(y_true, y_pred)[0] * std(y_true) * std(y_pred) /
                     (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred)) ** 2))

    def build_linear_model(self,
                           training_dataset: DataFrame,
                           testing_dataset: DataFrame,
                           panel_id: str,
                           params: list):
        # since some filter rules, like "known", performed bad, they were removed from the last time of simulation. 
        global driver_gene_set

        _method: str = panel_id[:-1] if panel_id[-1] in {'a', 'b', 'c', 'd', 'e'} else panel_id

        _recall, _precision, _gene_content, _hotspot, _vaf_cutoff = params

        _rgbrp: float = (1 / _recall) - (1 / _precision)
        _filter_string: str = 'type in @_gene_content & vaf >= @_vaf_cutoff'

        if _hotspot:
            _filter_string += ' & cancerhotspot == False & civic == False & cosmic96_coding < 20'
        else:
            pass

        _nonsense: bool = 'Nonsense_Mutation' in _gene_content
        _nonstop: bool = 'Nonstop_Mutation' in _gene_content
        _synonymous: bool = 'Synonymous_Mutation' in _gene_content
        _silent: bool = 'Silent' in _gene_content
        _splice_site: bool = 'Splice_Site' in _gene_content
        _translation_start_site: bool = 'Translation_Start_Site' in _gene_content

        # calculate the psTMB
        _tr: DataFrame = training_dataset.query(_filter_string).copy(deep=True)
        _te: DataFrame = testing_dataset.query(_filter_string).copy(deep=True)

        if _tr.empty or _te.empty:
            return []

        _tr['psTMB'] = (_tr.groupby(['Tumor_Sample_Barcode'])
                           .Tumor_Sample_Barcode
                           .transform('count') / self.bed_length)
        _te['psTMB'] = (_te.groupby(['Tumor_Sample_Barcode'])
                           .Tumor_Sample_Barcode
                           .transform('count') * (1 + _rgbrp) / self.bed_length)

        # build linear model
        _lr_md: LinearRegression = LinearRegression().fit(_tr['psTMB'].values.reshape(-1, 1),  # type: ignore
                                                          _tr['wesTMB'].values.reshape(-1, 1))  # type: ignore
        _coef: float64 = _lr_md.coef_[0][0]  # type: ignore
        _intercept: float64 = _lr_md.intercept_[0]  # type: ignore

        # test linear model
        _te['fTMB'] = maximum(_te['psTMB'] * _coef + _intercept, 0)  # type: ignore
        _te['frate'] = _te['fTMB'] >= 10.0
        _whfh_count: int = len(_te[(_te['rate']) & (_te['frate'])])
        _whfl_count: int = len(_te[(_te['rate']) & (~_te['frate'])])
        _wlfh_count: int = len(_te[(~_te['rate']) & (_te['frate'])])
        _wlfl_count: int = len(_te[(~_te['rate']) & (~_te['frate'])])

        # score linear model
        _te.drop_duplicates(subset=['Tumor_Sample_Barcode'], inplace=True)
        _r2: float = _lr_md.score(_te['psTMB'].values.reshape(-1, 1),  # type: ignore
                                  _te['wesTMB'].values.reshape(-1, 1))  # type: ignore
        _rmsle: float = mean_squared_log_error(_te['wesTMB'], _te['fTMB'], squared=False)  # type: ignore
        _acc: float = (_whfh_count + _wlfl_count) / (_whfh_count + _wlfl_count + _whfl_count + _wlfh_count)

        _te.drop_duplicates(subset=['fTMB', 'wesTMB'], inplace=True)
        _ccc: float = self.calculate_ccc(_te['wesTMB'], _te['fTMB'])

        return [panel_id, _method, self.bed_length,
                _recall, _precision,
                _nonsense, _nonstop, _synonymous, _silent, _splice_site, _translation_start_site,
                _hotspot, _vaf_cutoff,
                _coef, _intercept,
                _whfh_count, _wlfl_count, _whfl_count, _wlfh_count,
                _r2, _rmsle, _ccc, _acc]


def run_once(bed_obj: BedFilePreset, tr: DataFrame, te: DataFrame, param_once: list):
    _result: list = bed_obj.build_linear_model(training_dataset=tr,
                                               testing_dataset=te,
                                               panel_id=panel_id_dict[bed_obj.name],
                                               params=param_once)
    return _result


if __name__ == '__main__':
    tr_file = './mc3/fixed_mini_training_validation_set.parquet'
    te_file = './mc3/fixed_mini_testing_set.parquet'
    panel_id_file = './panel_id.xlsx'  # convert the panel ID from EQA ID to article ID.
    param_pkf = './params.pkl'
    bed_dir = './merged_somatic_bed'
    result_dir = './per_lab'

    with open(param_pkf, 'rb') as pkf:
        params_list: list = load(pkf)

    panel_id_df: DataFrame = read_excel(panel_id_file, sheet_name='tmb_parameters',
                                        usecols='A,B,C,W,X,Y', index_col=0)
    panel_id_dict: dict[str, str] = dict(zip(panel_id_df.index, panel_id_df['newID']))
    unique_panel_list: list = (panel_id_df[~panel_id_df['newID'].str.contains('[bcde]',
                                                                               regex=True)].index.tolist())

    tr_bedtool: BedTool = parquet2bedtool(tr_file)
    te_bedtool: BedTool = parquet2bedtool(te_file)

    total_bed_list: list[str] = get_bed_files(bed_dir, unique_panel_list)

    # panel_bed = total_bed_list[0]
    # panel_id: str = panel_bed.split('/')[-1].split('_')[0]
    # lab_bedfile = BedFilePreset(panel_bed)
    # lab_tr: DataFrame = lab_bedfile.get_dataset(tr_bedtool)
    # lab_te: DataFrame = lab_bedfile.get_dataset(te_bedtool)
    # print(run_once(lab_bedfile, lab_tr, lab_te, params_list[0]))

    for panel_bed in tqdm(total_bed_list, position=0, desc='panels'):
        panel_id: str = panel_bed.split('/')[-1].split('_')[0]
        print(panel_id)
        lab_bedfile = BedFilePreset(panel_bed)
        lab_tr: DataFrame = lab_bedfile.get_dataset(tr_bedtool)
        lab_te: DataFrame = lab_bedfile.get_dataset(te_bedtool)
        with Pool(processes=120) as pool:
            one_panel_results: list[list] = pool.starmap(run_once,
                                                         tqdm(zip(repeat(lab_bedfile),
                                                                  repeat(lab_tr),
                                                                  repeat(lab_te),
                                                                  params_list),
                                                              total=len(params_list),
                                                              position=1,
                                                              desc='params records',
                                                              leave=False))
            one_panel_results = [_r for _r in one_panel_results if _r]
        with open(f'{result_dir}/{panel_id}.pkl', 'wb+') as pkf2:
            dump(one_panel_results, pkf2)
        #  cleanup(remove_all=True)
