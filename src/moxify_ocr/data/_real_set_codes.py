# ruff: noqa: SIM905
# (1031-entry list literal vs whitespace-split string: the latter is the
# only readable form for this volume of data; SIM905 doesn't apply.)
"""Real Magic set codes pulled from Scryfall's sets.json.

Generated from ``data/scryfall/sets.json`` (1031 unique codes as of
the most recent ingest). Used by ``cardconjurer_specs`` to draw the majority
of synthetic samples from real codes (so the model's character distribution
matches what it sees in production) while still mixing in random codes for
generalization to future Wizards sets.

DO NOT edit by hand. Regenerate with:

    python -c "
    import json; from collections import Counter; from pathlib import Path
    data = json.load(open('data/scryfall/sets.json'))
    codes = sorted({s['code'].upper() for s in data['data'] if s.get('code')})
    # ... (see scripts/_regenerate_real_set_codes.py if we end up needing this often)
    "
"""

from __future__ import annotations

#: Tuple of every real Scryfall set code as of the last regeneration. Codes
#: are uppercase, lengths range 3-6 (most are 3 or 4), and ~33% contain a
#: digit (e.g. M21, 10E, 2X2).
REAL_SET_CODES: tuple[str, ...] = tuple(
    """\
10E 2ED 2X2 2XM 30A 3ED 40K 4BB 4ED 5DN
5ED 6ED 7ED 8ED 9ED A25 AA1 AA2 AA3 AA4
AACR AAFR ABLB ABRO ACLB ACMM ACR ADFT ADMU ADSK
AECL AEOE AER AFC AFDN AFIC AFIN AFR AINR AJMP
AKH AKHM AKR ALA ALCI ALL ALTC ALTR AMH1 AMH2
AMH3 AMID AMKM AMOM ANA ANB ANEO AONE AOTJ APC
ARB ARC ARN ASNC ASPM ASTX ATDM ATH ATLA ATLE
ATQ AVOW AVR AWOE AZNR BBD BCHR BFZ BIG BLB
BLC BNG BOK BOT BRB BRC BRO BRR BTD C13
C14 C15 C16 C17 C18 C19 C20 C21 CC1 CC2
CED CEI CHK CHR CLB CLU CM1 CM2 CMA CMB1
CMB2 CMD CMM CMR CN2 CNS CON CP1 CP2 CP3
CSP CST DBL DCI DD1 DD2 DDC DDD DDE DDF
DDG DDH DDI DDJ DDK DDL DDM DDN DDO DDP
DDQ DDR DDS DDT DDU DFT DGM DIS DKA DKM
DMC DMR DMU DOM DPA DRB DRC DRK DSC DSK
DST DTK DVD E01 E02 EA1 EA2 EA3 ECC ECL
ELD EMA EMN EOC EOE EOS EVE EVG EXO EXP
F01 F02 F03 F04 F05 F06 F07 F08 F09 F10
F11 F12 F13 F14 F15 F16 F17 F18 FBB FBRO
FCA FCLU FDC FDMU FDN FEM FFDN FIC FIN FJ22
FJ25 FJMP FLTR FMOM FNM FONE FRF FTLA FTMC FUT
G00 G01 G02 G03 G04 G05 G06 G07 G08 G09
G10 G11 G17 G18 G99 GDY GK1 GK2 GN2 GN3
GNT GPT GRN GS1 GTC GVL H09 H17 H1R H2R
HA1 HA2 HA3 HA4 HA5 HA6 HA7 HBG HHO HML
HOP HOU ICE IKO IMA INR INV ISD ITP J12
J13 J14 J15 J16 J17 J18 J19 J20 J21 J22
J25 JGP JMP JOU JP1 JTLA JUD JVC KHC KHM
KLD KLR KTK L12 L13 L14 L15 L16 L17 LCC
LCI LEA LEB LEG LGN LMAR LRW LTC LTR M10
M11 M12 M13 M14 M15 M19 M20 M21 M3C MACR
MAFR MAR MAT MB2 MBRO MBS MCLB MD1 MDMU ME1
ME2 ME3 ME4 MED MGB MH1 MH2 MH3 MIC MID
MIR MKC MKHM MKM MLTR MM2 MM3 MMA MMH2 MMID
MMQ MNEO MOC MOM MONE MOR MP2 MPR MPS MRD
MSC MSH MSNC MSTX MUL MVOW MZNR NCC NEC NEM
NEO NPH O90P OAFC OAFR OANA OARC OC13 OC14 OC15
OC16 OC17 OC18 OC19 OC20 OC21 OCLB OCM1 OCMD ODY
OE01 OGW OHOP OLEP OLGC OM1 OM2 OMB OMIC ONC
ONE ONS OPC2 OPCA ORI OTC OTJ OTP OVNT OVOC
P02 P03 P04 P05 P06 P07 P08 P09 P10 P10E
P11 P15A P22 P23 P2HG P30A P30H P30M P30T P5DN
P8ED P9ED PA1 PAER PAFR PAKH PAL00 PAL01 PAL02 PAL03
PAL04 PAL05 PAL06 PAL99 PALA PALP PANA PAPC PARB PARL
PAST PAVR PBBD PBFZ PBIG PBLB PBNG PBOK PBRO PC2
PCA PCBB PCEL PCHK PCLB PCMD PCMP PCMR PCNS PCON
PCSP PCY PD2 PD3 PDFT PDGM PDIS PDKA PDMU PDOM
PDP10 PDP12 PDP13 PDP14 PDP15 PDRC PDSK PDST PDTK PDTP
PECL PELD PELP PEMN PEOE PEVE PEWK PEXO PF19 PF20
PF23 PF24 PF25 PF26 PFDN PFIN PFRF PFUT PGPT PGPX
PGRN PGRU PGTC PH17 PH18 PH19 PH20 PH21 PH22 PH23
PHEL PHOP PHOU PHPR PHTR PHUK PIDW PIKO PINV PIO
PIP PISD PJ21 PJAS PJJT PJOU PJSC PJSE PJUD PKHM
PKLD PKTK PL21 PL22 PL23 PL24 PL25 PL26 PLC PLCI
PLG20 PLG21 PLG22 PLG24 PLG25 PLGM PLGN PLNY PLRW PLS
PLST PLTC PLTR PM10 PM11 PM12 PM13 PM14 PM15 PM19
PM20 PM21 PMAT PMBS PMDA PMEI PMH1 PMH2 PMH3 PMIC
PMID PMKM PMMQ PMOA PMOM PMOR PMPS PMPS06 PMPS07 PMPS08
PMPS09 PMPS10 PMPS11 PMRD PNAT PNCC PNEM PNEO PNPH PODY
POGW PONE PONS POR PORI POTJ PPC1 PPCY PPLC PPLS
PPP1 PPRO PPTK PR2 PR23 PRAV PRCQ PRED PRIX PRM
PRNA PROE PRTR PRW2 PRWK PS11 PS14 PS15 PS16 PS17
PS18 PS19 PSAL PSCG PSDC PSDG PSHM PSNC PSOI PSOK
PSOM PSPL PSPM PSS1 PSS2 PSS3 PSS4 PSS5 PSSC PSTH
PSTX PSUS PSVC PTBRO PTC PTDM PTDMU PTG PTHB PTHS
PTK PTKDF PTLA PTMP PTOR PTSNC PTSP PTSR PUDS PULG
PUMA PUNH PUNK PURL PUSG PUST PVAN PVOW PW11 PW12
PW21 PW22 PW23 PW24 PW25 PW26 PWAR PWCS PWOE PWOR
PWOS PWWK PXLN PXTC PZ1 PZ2 PZA PZEN PZNR Q06
Q07 RAV REN REX RFIN RIN RIX RNA ROE RQS
RTR RVR S00 S99 SBRO SCD SCG SCH SHM SIR
SIS SKHM SLC SLCI SLD SLP SLU SLX SMH3 SMID
SMOM SNC SNEO SOA SOC SOI SOK SOM SOS SPE
SPG SPM SS1 SS2 SS3 SSTX STA STH STX SUM
SUNF SVOW SZNR T10E T2X2 T2XM T30A T40K TA25 TACR
TAER TAFC TAFR TAKH TALA TARB TAVR TBBD TBFZ TBIG
TBLB TBLC TBNG TBOT TBRC TBRO TBTH TC14 TC15 TC16
TC17 TC18 TC19 TC20 TC21 TCLB TCM2 TCMA TCMM TCMR
TCN2 TCNS TCON TD0 TD2 TDAG TDC TDD1 TDD2 TDDC
TDDD TDDE TDDF TDDG TDDH TDDI TDDJ TDDK TDDL TDDM
TDDS TDDT TDDU TDFT TDGM TDKA TDM TDMC TDMR TDMU
TDOM TDRC TDSC TDSK TDTK TDVD TE01 TECC TECL TELD
TEMA TEMN TEOC TEOE TEVE TEVG TFDN TFIC TFIN TFRF
TFTH TGK1 TGK2 TGN2 TGN3 TGRN TGTC TGVL THB THOU
THP1 THP2 THP3 THS TIKO TIMA TINR TISD TJOU TJVC
TKHC TKHM TKLD TKTK TLA TLCC TLCI TLE TLRW TLTC
TLTR TM10 TM11 TM12 TM13 TM14 TM15 TM19 TM20 TM21
TM3C TMBS TMC TMD1 TMED TMH1 TMH2 TMH3 TMIC TMID
TMKC TMKM TMM2 TMM3 TMMA TMOC TMOM TMOR TMP TMSH
TMT TMUL TNCC TNEC TNEO TNPH TOGW TONC TONE TOR
TORI TOTC TOTJ TOTP TPCA TPIP TPR TREX TRIX TRNA
TROE TRTR TRVR TSB TSCD TSHM TSNC TSOC TSOI TSOM
TSOS TSP TSPM TSR TSTX TTDC TTDM TTHB TTHS TTLA
TTLE TTMC TTMT TTSR TUGL TUMA TUND TUNF TUST TVOC
TVOW TWAR TWHO TWOC TWOE TWWK TXLN TZEN TZNC TZNR
UDS UGIN UGL ULG ULST UMA UND UNF UNH UNK
USG UST V09 V10 V11 V12 V13 V14 V15 V16
V17 VIS VMA VOC VOW W16 W17 WAR WC00 WC01
WC02 WC03 WC04 WC97 WC98 WC99 WDMU WFIN WHO WMC
WMKM WMOM WOC WOE WONE WOT WTH WWK WWOE XANA
XLN YBLB YBRO YDFT YDMU YDSK YECL YEOE YLCI YMID
YMKM YNEO YONE YOTJ YSNC YTDM YWOE ZEN ZNC ZNE
ZNR
""".split()
)


#: Length distribution of real codes — used to weight random-code length
#: sampling to match reality (real is ~43% 3-char, ~54% 4-char, rest 5-6).
REAL_LENGTH_FREQ: dict[int, int] = {3: 441, 4: 562, 5: 22, 6: 6}


#: Character frequency over all real codes (uppercase letters + digits).
#: Used for weighted-random character sampling so synthetic codes match
#: real letter/digit distributions (e.g. P/T/M are common, Q/Z are rare).
CHAR_FREQ: dict[str, int] = {'0': 88, '1': 164, '2': 118, '3': 48, '4': 29, '5': 30, '6': 20, '7': 20, '8': 15, '9': 29, 'A': 172, 'B': 80, 'C': 219, 'D': 214, 'E': 124, 'F': 87, 'G': 81, 'H': 93, 'I': 70, 'J': 40, 'K': 71, 'L': 131, 'M': 256, 'N': 120, 'O': 164, 'P': 368, 'Q': 8, 'R': 141, 'S': 167, 'T': 321, 'U': 56, 'V': 43, 'W': 65, 'X': 25, 'Y': 22, 'Z': 18}


#: Manifest "salvage" map: Scryfall stores some promo/memorabilia sets with
#: a prefixed code (FJMP, PONE, AECL, ...) but the cards print the parent
#: set's code (JMP, ONE, ECL). This lookup translates the manifest code to
#: the printed code for label generation, so the model trains against what
#: the OCR actually sees. Filtered to set_type in {promo, memorabilia} —
#: alchemy (Y*) codes are digital-only and left as-is.
PROMO_SALVAGE: dict[str, str] = {
    "AACR": "ACR",
    "AAFR": "AFR",
    "ABLB": "BLB",
    "ABRO": "BRO",
    "ACLB": "CLB",
    "ACMM": "CMM",
    "ADFT": "DFT",
    "ADMU": "DMU",
    "ADSK": "DSK",
    "AECL": "ECL",
    "AEOE": "EOE",
    "AFDN": "FDN",
    "AFIC": "FIC",
    "AFIN": "FIN",
    "AINR": "INR",
    "AKHM": "KHM",
    "ALCI": "LCI",
    "ALTC": "LTC",
    "ALTR": "LTR",
    "AMH1": "MH1",
    "AMH2": "MH2",
    "AMH3": "MH3",
    "AMID": "MID",
    "AMKM": "MKM",
    "AMOM": "MOM",
    "ANEO": "NEO",
    "AONE": "ONE",
    "AOTJ": "OTJ",
    "ASNC": "SNC",
    "ASTX": "STX",
    "ATDM": "TDM",
    "ATLA": "TLA",
    "ATLE": "TLE",
    "AVOW": "VOW",
    "AWOE": "WOE",
    "AZNR": "ZNR",
    "FBRO": "BRO",
    "FCLU": "CLU",
    "FDMU": "DMU",
    "FFDN": "FDN",
    "FJ22": "J22",
    "FJ25": "J25",
    "FJMP": "JMP",
    "FLTR": "LTR",
    "FMOM": "MOM",
    "FONE": "ONE",
    "FTLA": "TLA",
    "FTMC": "TMC",
    "JTLA": "TLA",
    "LMAR": "MAR",
    "OAFC": "AFC",
    "OAFR": "AFR",
    "OC13": "C13",
    "OC14": "C14",
    "OC15": "C15",
    "OC16": "C16",
    "OC17": "C17",
    "OC18": "C18",
    "OC19": "C19",
    "OC20": "C20",
    "OC21": "C21",
    "OCLB": "CLB",
    "OCM1": "CM1",
    "OCMD": "CMD",
    "OMIC": "MIC",
    "OVOC": "VOC",
    "P10E": "10E",
    "P30A": "30A",
    "P30H": "30A",
    "P30M": "30A",
    "P30T": "30A",
    "P5DN": "5DN",
    "P8ED": "8ED",
    "P9ED": "9ED",
    "PAER": "AER",
    "PAFR": "AFR",
    "PAKH": "AKH",
    "PALA": "ALA",
    "PAPC": "APC",
    "PARB": "ARB",
    "PAVR": "AVR",
    "PBBD": "BBD",
    "PBFZ": "BFZ",
    "PBIG": "BIG",
    "PBLB": "BLB",
    "PBNG": "BNG",
    "PBOK": "BOK",
    "PBRO": "BRO",
    "PCHK": "CHK",
    "PCLB": "CLB",
    "PCMD": "CMD",
    "PCMR": "CMR",
    "PCNS": "CNS",
    "PCON": "CON",
    "PCSP": "CSP",
    "PDFT": "DFT",
    "PDGM": "DGM",
    "PDIS": "DIS",
    "PDKA": "DKA",
    "PDMU": "DMU",
    "PDOM": "DOM",
    "PDSK": "DSK",
    "PDST": "DST",
    "PDTK": "DTK",
    "PECL": "ECL",
    "PELD": "ELD",
    "PEMN": "EMN",
    "PEOE": "EOE",
    "PEVE": "EVE",
    "PEXO": "EXO",
    "PFDN": "FDN",
    "PFIN": "FIN",
    "PFRF": "FRF",
    "PFUT": "FUT",
    "PGPT": "GPT",
    "PGRN": "GRN",
    "PGTC": "GTC",
    "PHEL": "AVR",
    "PHOP": "HOP",
    "PHOU": "HOU",
    "PIKO": "IKO",
    "PINV": "INV",
    "PISD": "ISD",
    "PJOU": "JOU",
    "PJUD": "JUD",
    "PKHM": "KHM",
    "PKLD": "KLD",
    "PKTK": "KTK",
    "PLCI": "LCI",
    "PLGN": "LGN",
    "PLRW": "LRW",
    "PLTC": "LTR",
    "PLTR": "LTR",
    "PM10": "M10",
    "PM11": "M11",
    "PM12": "M12",
    "PM13": "M13",
    "PM14": "M14",
    "PM15": "M15",
    "PM19": "M19",
    "PM20": "M20",
    "PM21": "M21",
    "PMAT": "MAT",
    "PMBS": "MBS",
    "PMH1": "MH1",
    "PMH2": "MH2",
    "PMH3": "MH3",
    "PMIC": "PAST",
    "PMID": "MID",
    "PMKM": "MKM",
    "PMMQ": "MMQ",
    "PMOM": "MOM",
    "PMOR": "MOR",
    "PMRD": "MRD",
    "PNCC": "NCC",
    "PNEM": "NEM",
    "PNEO": "NEO",
    "PNPH": "NPH",
    "PODY": "ODY",
    "POGW": "OGW",
    "PONE": "ONE",
    "PONS": "ONS",
    "PORI": "ORI",
    "POTJ": "OTJ",
    "PPC1": "M15",
    "PPCY": "PCY",
    "PPLC": "PLC",
    "PPLS": "PLS",
    "PPP1": "M20",
    "PPTK": "PTK",
    "PRAV": "RAV",
    "PRIX": "RIX",
    "PRNA": "RNA",
    "PROE": "ROE",
    "PRTR": "RTR",
    "PRW2": "RNA",
    "PRWK": "GRN",
    "PSCG": "SCG",
    "PSHM": "SHM",
    "PSNC": "SNC",
    "PSOI": "SOI",
    "PSOK": "SOK",
    "PSOM": "SOM",
    "PSPM": "SPM",
    "PSS1": "BFZ",
    "PSS2": "XLN",
    "PSS3": "M19",
    "PSS4": "MKM",
    "PSS5": "FIN",
    "PSSC": "SLD",
    "PSTH": "STH",
    "PSTX": "STX",
    "PTDM": "TDM",
    "PTHB": "THB",
    "PTHS": "THS",
    "PTKDF": "DTK",
    "PTLA": "TLA",
    "PTMP": "TMP",
    "PTOR": "TOR",
    "PTSNC": "PSNC",
    "PTSP": "TSP",
    "PTSR": "TSR",
    "PUDS": "UDS",
    "PULG": "ULG",
    "PUNH": "UNH",
    "PUSG": "USG",
    "PUST": "UST",
    "PVOW": "VOW",
    "PWAR": "WAR",
    "PWOE": "WOE",
    "PWWK": "WWK",
    "PXLN": "XLN",
    "PXTC": "XLN",
    "PZEN": "ZEN",
    "PZNR": "ZNR",
    "RFIN": "FIN",
    "SLP": "SLD",
    "TBTH": "BNG",
    "TDAG": "JOU",
    "TFTH": "THS",
    "THP1": "THS",
    "THP2": "BNG",
    "THP3": "JOU",
    "UGIN": "FRF",
    "XANA": "ANA",
}
