from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource, SERVER_URL


"""

chembl-AC50-CHEMBL1741322/chembl-x-0.npy 1190
chembl-ALB-CHEMBL3885882/chembl-x-0.npy 1096
chembl-ALP-CHEMBL3885882/chembl-x-0.npy 1096
chembl-ALT-CHEMBL3885882/chembl-x-0.npy 1096
chembl-AST-CHEMBL3885882/chembl-x-0.npy 1096
chembl-BASOLE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-BILI-CHEMBL3885882/chembl-x-0.npy 1093
chembl-BUN-CHEMBL3885882/chembl-x-0.npy 1096
chembl-CHLORIDE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-CHOL-CHEMBL3885882/chembl-x-0.npy 1096
chembl-CK-CHEMBL3885882/chembl-x-0.npy 1014
chembl-CREAT-CHEMBL3885882/chembl-x-0.npy 1096
chembl-EOSLE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-GI50-CHEMBL1963844/chembl-x-0.npy 5049
chembl-GI50-CHEMBL1963848/chembl-x-0.npy 1900
chembl-GI50-CHEMBL1963854/chembl-x-0.npy 5503
chembl-GI50-CHEMBL1963860/chembl-x-0.npy 4757
chembl-GI50-CHEMBL1963866/chembl-x-0.npy 5068
chembl-GI50-CHEMBL1963868/chembl-x-0.npy 5344
chembl-GI50-CHEMBL1963874/chembl-x-0.npy 5419
chembl-GI50-CHEMBL1963876/chembl-x-0.npy 5377
chembl-GI50-CHEMBL1963880/chembl-x-0.npy 5340
chembl-GI50-CHEMBL1963882/chembl-x-0.npy 3596
chembl-GI50-CHEMBL1963885/chembl-x-0.npy 3578
chembl-GI50-CHEMBL1963887/chembl-x-0.npy 2020
chembl-GI50-CHEMBL1963889/chembl-x-0.npy 1859
chembl-GI50-CHEMBL1963895/chembl-x-0.npy 5167
chembl-GI50-CHEMBL1963900/chembl-x-0.npy 3311
chembl-GI50-CHEMBL1963901/chembl-x-0.npy 5282
chembl-GI50-CHEMBL1963903/chembl-x-0.npy 5344
chembl-GI50-CHEMBL1963911/chembl-x-0.npy 5004
chembl-GI50-CHEMBL1963921/chembl-x-0.npy 3570
chembl-GI50-CHEMBL1963922/chembl-x-0.npy 5438
chembl-GI50-CHEMBL1963929/chembl-x-0.npy 5404
chembl-GI50-CHEMBL1963935/chembl-x-0.npy 1541
chembl-GI50-CHEMBL1963945/chembl-x-0.npy 3327
chembl-GI50-CHEMBL1963953/chembl-x-0.npy 5249
chembl-GI50-CHEMBL1963954/chembl-x-0.npy 1925
chembl-GI50-CHEMBL1963960/chembl-x-0.npy 5379
chembl-GI50-CHEMBL1963961/chembl-x-0.npy 5187
chembl-GI50-CHEMBL1963963/chembl-x-0.npy 1988
chembl-GI50-CHEMBL1963976/chembl-x-0.npy 3484
chembl-GI50-CHEMBL1963981/chembl-x-0.npy 5430
chembl-GI50-CHEMBL1963985/chembl-x-0.npy 5458
chembl-GI50-CHEMBL1963989/chembl-x-0.npy 5332
chembl-GI50-CHEMBL1963990/chembl-x-0.npy 5364
chembl-GI50-CHEMBL1963991/chembl-x-0.npy 4645
chembl-GI50-CHEMBL1963994/chembl-x-0.npy 4968
chembl-GI50-CHEMBL1964004/chembl-x-0.npy 4985
chembl-GI50-CHEMBL1964006/chembl-x-0.npy 5460
chembl-GI50-CHEMBL1964007/chembl-x-0.npy 5377
chembl-GI50-CHEMBL1964009/chembl-x-0.npy 4883
chembl-GI50-CHEMBL1964012/chembl-x-0.npy 5351
chembl-GI50-CHEMBL1964014/chembl-x-0.npy 1561
chembl-GI50-CHEMBL1964017/chembl-x-0.npy 3558
chembl-GI50-CHEMBL1964018/chembl-x-0.npy 5020
chembl-GI50-CHEMBL1964021/chembl-x-0.npy 5356
chembl-GI50-CHEMBL1964025/chembl-x-0.npy 5053
chembl-GI50-CHEMBL1964030/chembl-x-0.npy 5379
chembl-GI50-CHEMBL1964034/chembl-x-0.npy 4679
chembl-GI50-CHEMBL1964037/chembl-x-0.npy 5222
chembl-GI50-CHEMBL1964040/chembl-x-0.npy 5354
chembl-GI50-CHEMBL1964043/chembl-x-0.npy 5271
chembl-GI50-CHEMBL1964045/chembl-x-0.npy 3639
chembl-GI50-CHEMBL1964047/chembl-x-0.npy 5487
chembl-GI50-CHEMBL1964048/chembl-x-0.npy 5460
chembl-GI50-CHEMBL1964049/chembl-x-0.npy 5266
chembl-GI50-CHEMBL1964059/chembl-x-0.npy 5396
chembl-GI50-CHEMBL1964062/chembl-x-0.npy 1905
chembl-GI50-CHEMBL1964063/chembl-x-0.npy 3494
chembl-GI50-CHEMBL1964065/chembl-x-0.npy 4421
chembl-GI50-CHEMBL1964066/chembl-x-0.npy 5309
chembl-GI50-CHEMBL1964072/chembl-x-0.npy 5503
chembl-GI50-CHEMBL1964074/chembl-x-0.npy 2025
chembl-GI50-CHEMBL1964075/chembl-x-0.npy 4893
chembl-GI50-CHEMBL1964077/chembl-x-0.npy 5423
chembl-GI50-CHEMBL1964085/chembl-x-0.npy 5306
chembl-GI50-CHEMBL1964086/chembl-x-0.npy 1659
chembl-GI50-CHEMBL1964087/chembl-x-0.npy 5325
chembl-GI50-CHEMBL1964088/chembl-x-0.npy 4984
chembl-GI50-CHEMBL1964091/chembl-x-0.npy 5150
chembl-GI50-CHEMBL1964092/chembl-x-0.npy 5080
chembl-GI50-CHEMBL1964099/chembl-x-0.npy 3056
chembl-GLUC-CHEMBL3885882/chembl-x-0.npy 1096
chembl-HCT-CHEMBL3885882/chembl-x-0.npy 1096
chembl-HGB-CHEMBL3885882/chembl-x-0.npy 1093
chembl-INHIBITION-CHEMBL4513217/chembl-x-0.npy 4807
chembl-INHIBITION-CHEMBL4513218/chembl-x-0.npy 4815
chembl-INHIBITION-CHEMBL4513219/chembl-x-0.npy 4815
chembl-INHIBITION-CHEMBL4513220/chembl-x-0.npy 4589
chembl-INHIBITION-CHEMBL4513221/chembl-x-0.npy 4815
chembl-Inhibition-CHEMBL3507681/chembl-x-0.npy 4574
chembl-Inhibition-CHEMBL3988443/chembl-x-0.npy 4574
chembl-Inhibition-CHEMBL4296187/chembl-x-0.npy 10334
chembl-Inhibition-CHEMBL4296188/chembl-x-0.npy 9777
chembl-Inhibition-CHEMBL4296802/chembl-x-0.npy 9464
chembl-Inhibition-CHEMBL4495582/chembl-x-0.npy 1433
chembl-Inhibition-CHEMBL4513082/chembl-x-0.npy 1433
chembl-LYMLE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-MCH-CHEMBL3885882/chembl-x-0.npy 1093
chembl-MCHC-CHEMBL3885882/chembl-x-0.npy 1093
chembl-MCV-CHEMBL3885882/chembl-x-0.npy 1096
chembl-MONOLE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-NEUTLE-CHEMBL3885882/chembl-x-0.npy 1096
chembl-PHOS-CHEMBL3885882/chembl-x-0.npy 1096
chembl-PLAT-CHEMBL3885882/chembl-x-0.npy 1096
chembl-POTASSIUM-CHEMBL3885882/chembl-x-0.npy 1096
chembl-PROT-CHEMBL3885882/chembl-x-0.npy 1096
chembl-Potency-CHEMBL1613836/chembl-x-0.npy 4118
chembl-Potency-CHEMBL1613838/chembl-x-0.npy 3004
chembl-Potency-CHEMBL1613842/chembl-x-0.npy 3090
chembl-Potency-CHEMBL1613910/chembl-x-0.npy 1107
chembl-Potency-CHEMBL1613914/chembl-x-0.npy 3668
chembl-Potency-CHEMBL1613918/chembl-x-0.npy 1278
chembl-Potency-CHEMBL1613970/chembl-x-0.npy 1009
chembl-Potency-CHEMBL1614038/chembl-x-0.npy 2568
chembl-Potency-CHEMBL1614076/chembl-x-0.npy 1239
chembl-Potency-CHEMBL1614079/chembl-x-0.npy 1402
chembl-Potency-CHEMBL1614087/chembl-x-0.npy 4682
chembl-Potency-CHEMBL1614146/chembl-x-0.npy 4118
chembl-Potency-CHEMBL1614161/chembl-x-0.npy 2131
chembl-Potency-CHEMBL1614166/chembl-x-0.npy 2027
chembl-Potency-CHEMBL1614174/chembl-x-0.npy 2482
chembl-Potency-CHEMBL1614211/chembl-x-0.npy 3493
chembl-Potency-CHEMBL1614227/chembl-x-0.npy 1200
chembl-Potency-CHEMBL1614236/chembl-x-0.npy 3023
chembl-Potency-CHEMBL1614249/chembl-x-0.npy 3282
chembl-Potency-CHEMBL1614250/chembl-x-0.npy 1111
chembl-Potency-CHEMBL1614257/chembl-x-0.npy 1739
chembl-Potency-CHEMBL1614275/chembl-x-0.npy 1887
chembl-Potency-CHEMBL1614280/chembl-x-0.npy 1240
chembl-Potency-CHEMBL1614281/chembl-x-0.npy 1155
chembl-Potency-CHEMBL1614342/chembl-x-0.npy 2517
chembl-Potency-CHEMBL1614361/chembl-x-0.npy 1531
chembl-Potency-CHEMBL1614364/chembl-x-0.npy 1195
chembl-Potency-CHEMBL1614410/chembl-x-0.npy 1093
chembl-Potency-CHEMBL1614421/chembl-x-0.npy 2639
chembl-Potency-CHEMBL1614441/chembl-x-0.npy 3371
chembl-Potency-CHEMBL1614458/chembl-x-0.npy 6304
chembl-Potency-CHEMBL1614459/chembl-x-0.npy 7096
chembl-Potency-CHEMBL1614530/chembl-x-0.npy 4849
chembl-Potency-CHEMBL1614544/chembl-x-0.npy 2218
chembl-Potency-CHEMBL1737902/chembl-x-0.npy 6480
chembl-Potency-CHEMBL1737991/chembl-x-0.npy 4048
chembl-Potency-CHEMBL1738132/chembl-x-0.npy 2610
chembl-Potency-CHEMBL1738184/chembl-x-0.npy 4994
chembl-Potency-CHEMBL1738312/chembl-x-0.npy 3919
chembl-Potency-CHEMBL1738317/chembl-x-0.npy 3331
chembl-Potency-CHEMBL1738442/chembl-x-0.npy 7331
chembl-Potency-CHEMBL1738588/chembl-x-0.npy 7848
chembl-Potency-CHEMBL1738606/chembl-x-0.npy 1208
chembl-Potency-CHEMBL1794308/chembl-x-0.npy 3654
chembl-Potency-CHEMBL1794311/chembl-x-0.npy 1799
chembl-Potency-CHEMBL1794345/chembl-x-0.npy 13702
chembl-Potency-CHEMBL1794352/chembl-x-0.npy 5877
chembl-Potency-CHEMBL1794359/chembl-x-0.npy 1537
chembl-Potency-CHEMBL1794375/chembl-x-0.npy 6607
chembl-Potency-CHEMBL1794401/chembl-x-0.npy 8322
chembl-Potency-CHEMBL1794424/chembl-x-0.npy 2448
chembl-Potency-CHEMBL1794440/chembl-x-0.npy 1002
chembl-Potency-CHEMBL1794461/chembl-x-0.npy 1596
chembl-Potency-CHEMBL1794483/chembl-x-0.npy 9530
chembl-Potency-CHEMBL1794499/chembl-x-0.npy 1290
chembl-Potency-CHEMBL1794553/chembl-x-0.npy 3546
chembl-Potency-CHEMBL1794580/chembl-x-0.npy 9915
chembl-Potency-CHEMBL1794584/chembl-x-0.npy 3014
chembl-Potency-CHEMBL1794585/chembl-x-0.npy 1200
chembl-Potency-CHEMBL2114713/chembl-x-0.npy 1250
chembl-Potency-CHEMBL2114738/chembl-x-0.npy 1580
chembl-Potency-CHEMBL2114775/chembl-x-0.npy 3350
chembl-Potency-CHEMBL2114780/chembl-x-0.npy 4186
chembl-Potency-CHEMBL2114784/chembl-x-0.npy 2479
chembl-Potency-CHEMBL2114788/chembl-x-0.npy 5101
chembl-Potency-CHEMBL2114807/chembl-x-0.npy 2307
chembl-Potency-CHEMBL2114810/chembl-x-0.npy 7269
chembl-Potency-CHEMBL2114836/chembl-x-0.npy 1881
chembl-Potency-CHEMBL2114843/chembl-x-0.npy 4050
chembl-Potency-CHEMBL2114861/chembl-x-0.npy 1798
chembl-Potency-CHEMBL2114908/chembl-x-0.npy 1139
chembl-Potency-CHEMBL2114913/chembl-x-0.npy 2068
chembl-Potency-CHEMBL2354211/chembl-x-0.npy 1594
chembl-Potency-CHEMBL2354221/chembl-x-0.npy 6549
chembl-Potency-CHEMBL2354254/chembl-x-0.npy 5683
chembl-Potency-CHEMBL2354287/chembl-x-0.npy 3305
chembl-Potency-CHEMBL2354311/chembl-x-0.npy 1538
chembl-Potency-CHEMBL3214953/chembl-x-0.npy 1102
chembl-Potency-CHEMBL3215017/chembl-x-0.npy 1197
chembl-Potency-CHEMBL3215106/chembl-x-0.npy 2289
chembl-Potency-CHEMBL3215181/chembl-x-0.npy 3089
chembl-Potency-CHEMBL3215278/chembl-x-0.npy 2233
chembl-Potency-CHEMBL3562077/chembl-x-0.npy 5146
chembl-RBC-CHEMBL3885882/chembl-x-0.npy 1096
chembl-SODIUM-CHEMBL3885882/chembl-x-0.npy 1096
chembl-WBC-CHEMBL3885882/chembl-x-0.npy 1095
chembl-WEIGHT-CHEMBL3885862/chembl-x-0.npy 3196
chembl-WEIGHT-CHEMBL3885863/chembl-x-0.npy 3360

"""


CHEMBL_FILES = ['chembl-AC50-CHEMBL1741322/chembl-x-0.npy',
                'chembl-ALB-CHEMBL3885882/chembl-x-0.npy',
                'chembl-ALP-CHEMBL3885882/chembl-x-0.npy',
                'chembl-ALT-CHEMBL3885882/chembl-x-0.npy',
                'chembl-AST-CHEMBL3885882/chembl-x-0.npy',
                'chembl-BASOLE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-BILI-CHEMBL3885882/chembl-x-0.npy',
                'chembl-BUN-CHEMBL3885882/chembl-x-0.npy',
                'chembl-CHLORIDE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-CHOL-CHEMBL3885882/chembl-x-0.npy',
                'chembl-CK-CHEMBL3885882/chembl-x-0.npy',
                'chembl-CREAT-CHEMBL3885882/chembl-x-0.npy',
                'chembl-EOSLE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963844/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963848/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963854/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963860/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963866/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963868/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963874/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963876/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963880/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963882/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963885/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963887/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963889/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963895/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963900/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963901/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963903/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963911/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963921/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963922/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963929/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963935/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963945/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963953/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963954/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963960/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963961/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963963/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963976/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963981/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963985/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963989/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963990/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963991/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1963994/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964004/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964006/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964007/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964009/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964012/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964014/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964017/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964018/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964021/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964025/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964030/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964034/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964037/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964040/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964043/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964045/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964047/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964048/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964049/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964059/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964062/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964063/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964065/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964066/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964072/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964074/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964075/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964077/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964085/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964086/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964087/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964088/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964091/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964092/chembl-x-0.npy',
                'chembl-GI50-CHEMBL1964099/chembl-x-0.npy',
                'chembl-GLUC-CHEMBL3885882/chembl-x-0.npy',
                'chembl-HCT-CHEMBL3885882/chembl-x-0.npy',
                'chembl-HGB-CHEMBL3885882/chembl-x-0.npy',
                'chembl-INHIBITION-CHEMBL4513217/chembl-x-0.npy',
                'chembl-INHIBITION-CHEMBL4513218/chembl-x-0.npy',
                'chembl-INHIBITION-CHEMBL4513219/chembl-x-0.npy',
                'chembl-INHIBITION-CHEMBL4513220/chembl-x-0.npy',
                'chembl-INHIBITION-CHEMBL4513221/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL3507681/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL3988443/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL4296187/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL4296188/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL4296802/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL4495582/chembl-x-0.npy',
                'chembl-Inhibition-CHEMBL4513082/chembl-x-0.npy',
                'chembl-LYMLE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-MCH-CHEMBL3885882/chembl-x-0.npy',
                'chembl-MCHC-CHEMBL3885882/chembl-x-0.npy',
                'chembl-MCV-CHEMBL3885882/chembl-x-0.npy',
                'chembl-MONOLE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-NEUTLE-CHEMBL3885882/chembl-x-0.npy',
                'chembl-PHOS-CHEMBL3885882/chembl-x-0.npy',
                'chembl-PLAT-CHEMBL3885882/chembl-x-0.npy',
                'chembl-POTASSIUM-CHEMBL3885882/chembl-x-0.npy',
                'chembl-PROT-CHEMBL3885882/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613836/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613838/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613842/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613910/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613914/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613918/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1613970/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614038/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614076/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614079/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614087/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614146/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614161/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614166/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614174/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614211/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614227/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614236/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614249/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614250/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614257/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614275/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614280/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614281/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614342/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614361/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614364/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614410/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614421/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614441/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614458/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614459/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614530/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1614544/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1737902/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1737991/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738132/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738184/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738312/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738317/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738442/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738588/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1738606/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794308/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794311/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794345/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794352/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794359/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794375/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794401/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794424/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794440/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794461/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794483/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794499/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794553/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794580/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794584/chembl-x-0.npy',
                'chembl-Potency-CHEMBL1794585/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114713/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114738/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114775/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114780/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114784/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114788/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114807/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114810/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114836/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114843/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114861/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114908/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2114913/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2354211/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2354221/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2354254/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2354287/chembl-x-0.npy',
                'chembl-Potency-CHEMBL2354311/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3214953/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3215017/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3215106/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3215181/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3215278/chembl-x-0.npy',
                'chembl-Potency-CHEMBL3562077/chembl-x-0.npy',
                'chembl-RBC-CHEMBL3885882/chembl-x-0.npy',
                'chembl-SODIUM-CHEMBL3885882/chembl-x-0.npy',
                'chembl-WBC-CHEMBL3885882/chembl-x-0.npy',
                'chembl-WEIGHT-CHEMBL3885862/chembl-x-0.npy',
                'chembl-WEIGHT-CHEMBL3885863/chembl-x-0.npy']


class ChEMBLDataset(DiscreteDataset):
    """A molecule design dataset that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
    x_name: str
        An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper
    y_name: str
        An attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

    x: np.ndarray
        the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type
    input_shape: Tuple[int]
        the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape
    input_size: int
        the total number of components in the design values 'x', represented
        as a single integer, the product of its shape entries
    input_dtype: np.dtype
        the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

    y: np.ndarray
        the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'
    output_shape: Tuple[int]
        the shape of a single prediction value 'y', represented as a list of
        integers similar to calling np.ndarray.shape
    output_size: int
        the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries
    output_dtype: np.dtype
        the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

    dataset_size: int
        the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer
    dataset_distribution: Callable[np.ndarray, np.ndarray]
        the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution
    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class
    dataset_max_output: float
        the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_output: float
        the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

    internal_batch_size: int
        the integer number of samples per batch that is used internally
        when processing the dataset and generating samples
    freeze_statistics: bool
        a boolean indicator that when set to true prevents methods from
        changing the normalization and sub sampling statistics

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values
        in the dataset are being normalized
    x_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible design values in the dataset
    x_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible design values in the dataset

    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized
    y_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible prediction values in the dataset
    y_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible prediction values in the dataset

    is_logits: bool (only supported for a DiscreteDataset)
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

    Public Methods:

    iterate_batches(batch_size: int, return_x: bool,
                    return_y: bool, drop_remainder: bool)
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model
    iterate_samples(return_x: bool, return_y: bool):
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

    subsample(max_samples: int,
              max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed
    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

    clone(subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        Generate a cloned copy of a model-based optimization dataset
        using the provided name and shard generation settings; useful
        when relabelling a dataset buffer from the dis
    split(fraction: float, subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        split a model-based optimization data set into a training set and
        a validation set allocating 'fraction' of the data set to the
        validation set and the rest to the training set

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardization so that they have their
        original empirical mean and variance
    normalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and undoes standardization so that they have their
        original empirical mean and variance

    map_normalize_x():
        a destructive function that standardizes the design values 'x'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_x():
        a destructive function that undoes standardization of the
        design values 'x' in the class dataset in-place which are expected
        to have zero  empirical mean and unit variance
    map_normalize_y():
        a destructive function that standardizes the prediction values 'y'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_y():
        a destructive function that undoes standardization of the
        prediction values 'y' in the class dataset in-place which are
        expected to have zero empirical mean and unit variance

    --- for discrete tasks only

    to_logits(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution
    to_integers(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

    map_to_logits():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits
    map_to_integers():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

    """

    name = "chembl/chembl"
    y_name = "standard_value"
    x_name = "smiles"

    @staticmethod
    def register_x_shards(assay_chembl_id="CHEMBL1794345",
                          standard_type="Potency"):  # max percentile 53 works well
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        assay_chembl_id: str
            a string identifier that specifies which assay to use for
            model-based optimization, where the goal is to find a design
            value 'x' that maximizes a certain property
        standard_type: str
            a string identifier that specifies which property of the assay
            is being measured for model-based optimization, where the goal is
            to maximize that property

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file, is_absolute=False,
            download_target=file,
            download_method="direct") for file in CHEMBL_FILES
            if f"{standard_type}-{assay_chembl_id}" in file]

    @staticmethod
    def register_y_shards(assay_chembl_id="CHEMBL1794345",
                          standard_type="Potency"):
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        assay_chembl_id: str
            a string identifier that specifies which assay to use for
            model-based optimization, where the goal is to find a design
            value 'x' that maximizes a certain property
        standard_type: str
            a string identifier that specifies which property of the assay
            is being measured for model-based optimization, where the goal is
            to maximize that property

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=file.replace('-x-', '-y-'),
            download_method="direct") for file in CHEMBL_FILES
            if f"{standard_type}-{assay_chembl_id}" in file]

    def __init__(self, assay_chembl_id="CHEMBL1794345",
                 standard_type="Potency",
                 soft_interpolation=0.6, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        assay_chembl_id: str
            a string identifier that specifies which assay to use for
            model-based optimization, where the goal is to find a design
            value 'x' that maximizes a certain property
        standard_type: str
            a string identifier that specifies which property of the assay
            is being measured for model-based optimization, where the goal is
            to maximize that property
        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # set the names the describe the dataset
        self.name = f"chembl-{standard_type}-{assay_chembl_id}/chembl"
        self.y_name = standard_type

        # initialize the dataset using the method in the base class
        super(ChEMBLDataset, self).__init__(
            self.register_x_shards(assay_chembl_id=assay_chembl_id,
                                   standard_type=standard_type),
            self.register_y_shards(assay_chembl_id=assay_chembl_id,
                                   standard_type=standard_type),
            is_logits=False, num_classes=591,
            soft_interpolation=soft_interpolation, **kwargs)
