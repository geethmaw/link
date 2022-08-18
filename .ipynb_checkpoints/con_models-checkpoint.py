# @Author: geethmawerapitiya
# @Date:   2022-07-14T11:25:14-06:00
# @Project: Research
# @Filename: con_models.py
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-15T15:17:17-06:00



def get_cons():
    #####Constants
    Cp = 1004           #J/kg/K
    Rd = 287            #J/kg/K
    con= Rd/Cp

    #use_colors = ['#88CCEE','#CC6677','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#661100','#6699CC','#888888','#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#808080','#ffffff','#000000','#000080','#d2f53c'] #'#ffe119', ,'#DDCC77'

    use_colors = ['red','goldenrod','teal','blue','hotpink','green','magenta','cornflowerblue','mediumpurple','deeppink','coral','burlywood','yellow','black','#88CCEE','#CC6677','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#661100','#6699CC','#888888','#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#808080','#ffffff','#000000','#000080','#d2f53c','blueviolet','rosybrown','cyan','lawngreen','tomato','peru','salmon']



    ########models

    modname = ['CESM2','CanESM5','CESM2-WACCM-FV2','GFDL-CM4','NorESM2-LM','CESM2-FV2','CESM2-WACCM','CMCC-CM2-HR4','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1','HadGEM3-GC31-LL','HadGEM3-GC31-MM','INM-CM4-8','INM-CM5-0','IPSL-CM5A2-INCA','IPSL-CM6A-LR','MPI-ESM-1-2-HAM','AWI-ESM-1-1-LR','MPI-ESM1-2-HR','MPI-ESM1-2-LR','NorESM2-MM','UKESM1-0-LL']



    warming_modname = ['CESM2','CESM2-FV2','CESM2-WACCM','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','INM-CM4-8','INM-CM5-0'] #'CMCC-CM2-HR4',

    hiresmd = ['BCC-CSM2-HR','FGOALS-f3-L','MRI-AGCM3-2-H'] #,'MRI-AGCM3-2-S'

    amip_md = ['CESM2','IPSL-CM6A-LR','GISS-E3-G']

    #######variables
    varname         = ['sfcWind', 'ts','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
    pvarname        = ['ta']


    return(con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md)
