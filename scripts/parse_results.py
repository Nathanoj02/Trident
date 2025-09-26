stdout = '''
----Running HnS-SpGEMM----
A matrix file path: /global/homes/t/thomasp/HnS-SpGEMM/matrices/known_squaring_nnz/vanHeukelum/cage15/cage15.bmtx
B matrix file path: /global/homes/t/thomasp/HnS-SpGEMM/matrices/known_squaring_nnz/vanHeukelum/cage15/cage15.bmtx
Number of processes per row: 2
Number of processes per col: 2
Number of processes per node: 1
Chosen implementation: main(main use MPI_Put)
A stored in CSC format: 1
Spcomm enabled: 0 (It require --Acsc)
         col 0          col 1          
       ------------------- ------------------- 
row 0  | Node0 [0  ]     | Node0 [1  ]     |
       ------------------- ------------------- 
row 1  | Node0 [2  ]     | Node0 [3  ]     |
       ------------------- ------------------- 
Beginning conversion
Done conversion
Beginning spgemm -- implementation: main
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=80.441933,stddev=39.232296,min=52.700512,max=108.183357,sum=160.883865
<[process 0]>[comp_time] n=2,avg=378.768280,stddev=91.374817,min=314.156525,max=443.380035,sum=757.536560
<[process 0]>[data_proc_A] n=2,avg=242.908447,stddev=341.222656,min=1.627584,max=484.189301,sum=485.816895
<[process 0]>[data_proc_B] n=2,avg=24.127888,stddev=33.386776,min=0.519872,max=47.735905,sum=48.255775
NNZ C: 929023247
<Timer>[spgemm] 2296.719482 ms
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=59.890045,stddev=40.993816,min=30.903040,max=88.877052,sum=119.780090
<[process 0]>[comp_time] n=2,avg=307.547180,stddev=106.434570,min=232.286560,max=382.807770,sum=615.094360
<[process 0]>[data_proc_A] n=2,avg=3.471328,stddev=2.371716,min=1.794272,max=5.148384,sum=6.942656
<[process 0]>[data_proc_B] n=2,avg=0.761968,stddev=0.299248,min=0.550368,max=0.973568,sum=1.523936
NNZ C: 929023247
<Timer>[spgemm] n=2,avg=1775.370850,stddev=737.298340,min=1254.022217,max=2296.719482,sum=3550.741699
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=16.104912,stddev=22.641558,min=0.094912,max=32.114910,sum=32.209824
<[process 0]>[comp_time] n=2,avg=270.779205,stddev=145.124146,min=168.160934,max=373.397461,sum=541.558411
<[process 0]>[data_proc_A] n=2,avg=8.634992,stddev=9.496885,min=1.919680,max=15.350304,sum=17.269983
<[process 0]>[data_proc_B] n=2,avg=2.755104,stddev=3.089095,min=0.570784,max=4.939424,sum=5.510208
NNZ C: 929023247
<Timer>[spgemm] n=3,avg=1640.472778,stddev=571.311523,min=1254.022217,max=2296.719482,sum=4921.418457
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=54.013649,stddev=56.964954,min=13.733344,max=94.293953,sum=108.027298
<[process 0]>[comp_time] n=2,avg=283.601959,stddev=133.693329,min=189.066498,max=378.137421,sum=567.203918
<[process 0]>[data_proc_A] n=2,avg=6.931072,stddev=1.256817,min=6.042368,max=7.819776,sum=13.862144
<[process 0]>[data_proc_B] n=2,avg=11.369008,stddev=14.397848,min=1.188192,max=21.549824,sum=22.738016
NNZ C: 929023247
<Timer>[spgemm] n=4,avg=1550.570557,stddev=499.927551,min=1254.022217,max=2296.719482,sum=6202.282227
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=38.982674,stddev=26.732870,min=20.079679,max=57.885666,sum=77.965347
<[process 0]>[comp_time] n=2,avg=274.013977,stddev=143.442642,min=172.584702,max=375.443237,sum=548.027954
<[process 0]>[data_proc_A] n=2,avg=44.122753,stddev=60.082943,min=1.637696,max=86.607811,sum=88.245506
<[process 0]>[data_proc_B] n=2,avg=34.280800,stddev=47.761547,min=0.508288,max=68.053314,sum=68.561600
NNZ C: 929023247
<Timer>[spgemm] n=5,avg=1486.759888,stddev=455.855988,min=1231.517212,max=2296.719482,sum=7433.799316
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=20.108288,stddev=16.439135,min=8.484064,max=31.732512,sum=40.216576
<[process 0]>[comp_time] n=2,avg=266.898682,stddev=151.643356,min=159.670624,max=374.126709,sum=533.797363
<[process 0]>[data_proc_A] n=2,avg=4.144320,stddev=3.412576,min=1.731264,max=6.557376,sum=8.288640
<[process 0]>[data_proc_B] n=2,avg=0.851632,stddev=0.381068,min=0.582176,max=1.121088,sum=1.703264
NNZ C: 929023247
<Timer>[spgemm] n=6,avg=1457.552368,stddev=413.959167,min=1231.517212,max=2296.719482,sum=8745.314453
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=32.141937,stddev=31.624598,min=9.779968,max=54.503902,sum=64.283875
<[process 0]>[comp_time] n=2,avg=273.090607,stddev=141.410278,min=173.098434,max=373.082764,sum=546.181213
<[process 0]>[data_proc_A] n=2,avg=6.866496,stddev=7.496599,min=1.565600,max=12.167392,sum=13.732992
<[process 0]>[data_proc_B] n=2,avg=2.562496,stddev=2.881963,min=0.524640,max=4.600352,sum=5.124992
NNZ C: 929023247
<Timer>[spgemm] n=7,avg=1436.398560,stddev=382.013367,min=1231.517212,max=2296.719482,sum=10054.790039
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=39.458096,stddev=39.812782,min=11.306208,max=67.609985,sum=78.916191
<[process 0]>[comp_time] n=2,avg=300.588318,stddev=95.699142,min=232.918823,max=368.257843,sum=601.176636
<[process 0]>[data_proc_A] n=2,avg=4.174528,stddev=3.482677,min=1.711904,max=6.637152,sum=8.349056
<[process 0]>[data_proc_B] n=2,avg=0.778896,stddev=0.478502,min=0.440544,max=1.117248,sum=1.557792
NNZ C: 929023247
<Timer>[spgemm] n=8,avg=1417.270386,stddev=357.789795,min=1231.517212,max=2296.719482,sum=11338.163086
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=30.363985,stddev=42.763851,min=0.125376,max=60.602592,sum=60.727970
<[process 0]>[comp_time] n=2,avg=296.348755,stddev=181.883698,min=167.737564,max=424.959961,sum=592.697510
<[process 0]>[data_proc_A] n=2,avg=7.157328,stddev=1.149405,min=6.344576,max=7.970080,sum=14.314655
<[process 0]>[data_proc_B] n=2,avg=3.233456,stddev=3.233254,min=0.947200,max=5.519712,sum=6.466912
NNZ C: 929023247
<Timer>[spgemm] n=9,avg=1406.351074,stddev=336.281067,min=1231.517212,max=2296.719482,sum=12657.159180
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 1]>[comm_wait] n=2,avg=685.595459,stddev=639.573914,min=233.348389,max=1137.842529,sum=1371.190918
<[process 1]>[comp_time] n=2,avg=86.120041,stddev=22.791489,min=70.004028,max=102.236061,sum=172.240082
<[process 1]>[data_proc_A] n=2,avg=13.595792,stddev=10.505254,min=6.167456,max=21.024128,sum=27.191584
<[process 1]>[data_proc_B] n=2,avg=1.167968,stddev=0.847849,min=0.568448,max=1.767488,sum=2.335936
<[process 1]>[comm_wait] n=2,avg=157.217667,stddev=18.998575,min=143.783646,max=170.651688,sum=314.435333
<[process 1]>[comp_time] n=2,avg=65.219887,stddev=51.930260,min=28.499647,max=101.940125,sum=130.439774
<[process 1]>[data_proc_A] n=2,avg=8.105856,stddev=4.086149,min=5.216512,max=10.995200,sum=16.211712
<[process 1]>[data_proc_B] n=2,avg=11.936768,stddev=15.357725,min=1.077216,max=22.796320,sum=23.873535
<[process 1]>[comm_wait] n=2,avg=202.439545,stddev=9.636909,min=195.625214,max=209.253860,sum=404.879089
<[process 1]>[comp_time] n=2,avg=87.350288,stddev=79.952232,min=30.815519,max=143.885056,sum=174.700577
<[process 1]>[data_proc_A] n=2,avg=5.499232,stddev=0.007512,min=5.493920,max=5.504544,sum=10.998464
<[process 1]>[data_proc_B] n=2,avg=1.011856,stddev=0.367085,min=0.752288,max=1.271424,sum=2.023712
<[process 1]>[comm_wait] n=2,avg=169.577408,stddev=61.645138,min=125.987709,max=213.167099,sum=339.154816
<[process 1]>[comp_time] n=2,avg=75.720963,stddev=63.699799,min=30.678400,max=120.763519,sum=151.441925
<[process 1]>[data_proc_A] n=2,avg=9.011440,stddev=6.494905,min=4.418848,max=13.604032,sum=18.022881
<[process 1]>[data_proc_B] n=2,avg=3.227744,stddev=2.814986,min=1.237248,max=5.218240,sum=6.455488
<[process 1]>[comm_wait] n=2,avg=127.238220,stddev=72.326393,min=76.095741,max=178.380707,sum=254.476440
<[process 1]>[comp_time] n=2,avg=81.722702,stddev=70.972908,min=31.537279,max=131.908127,sum=163.445404
<[process 1]>[data_proc_A] n=2,avg=7.616288,stddev=5.965583,min=3.397984,max=11.834592,sum=15.232576
<[process 1]>[data_proc_B] n=2,avg=2.834048,stddev=2.250749,min=1.242528,max=4.425568,sum=5.668096
<[process 1]>[comm_wait] n=2,avg=193.097656,stddev=6.569967,min=188.451996,max=197.743332,sum=386.195312
<[process 1]>[comp_time] n=2,avg=66.627678,stddev=52.561363,min=29.461184,max=103.794174,sum=133.255356
<[process 1]>[data_proc_A] n=2,avg=6.415024,stddev=1.015767,min=5.696768,max=7.133280,sum=12.830048
<[process 1]>[data_proc_B] n=2,avg=1.002640,stddev=0.299067,min=0.791168,max=1.214112,sum=2.005280
<[process 1]>[comm_wait] n=2,avg=202.631378,stddev=6.319001,min=198.163162,max=207.099579,sum=405.262756
<[process 1]>[comp_time] n=2,avg=76.129822,stddev=55.052235,min=37.202015,max=115.057632,sum=152.259644
<[process 1]>[data_proc_A] n=2,avg=8.471807,stddev=4.493172,min=5.294656,max=11.648960,sum=16.943615
<[process 1]>[data_proc_B] n=2,avg=4.756080,stddev=2.734727,min=2.822336,max=6.689824,sum=9.512160
<[process 1]>[comm_wait] n=2,avg=185.839630,stddev=64.973701,min=139.896286,max=231.782974,sum=371.679260
<[process 1]>[comp_time] n=2,avg=93.243568,stddev=89.675598,min=29.833344,max=156.653793,sum=186.487137
<[process 1]>[data_proc_A] n=2,avg=5.954928,stddev=0.094288,min=5.888256,max=6.021600,sum=11.909856
<[process 1]>[data_proc_B] n=2,avg=0.957136,stddev=0.508551,min=0.597536,max=1.316736,sum=1.914272
<[process 1]>[comm_wait] n=2,avg=121.724442,stddev=38.082081,min=94.796349,max=148.652542,sum=243.448883
<[process 1]>[comp_time] n=2,avg=93.816879,stddev=91.293419,min=29.262688,max=158.371078,sum=187.633759
<[process 1]>[data_proc_A] n=2,avg=7.345840,stddev=5.913245,min=3.164544,max=11.527136,sum=14.691680
<[process 1]>[data_proc_B] n=2,avg=2.883664,stddev=2.322410,min=1.241472,max=4.525856,sum=5.767328
<[process 2]>[comm_wait] n=2,avg=717.494568,stddev=600.930542,min=292.572510,max=1142.416626,sum=1434.989136
<[process 2]>[comp_time] n=2,avg=72.679695,stddev=39.945740,min=44.433792,max=100.925598,sum=145.359390
<[process 2]>[data_proc_A] n=2,avg=7.704400,stddev=8.012391,min=2.038784,max=13.370016,sum=15.408800
<[process 2]>[data_proc_B] n=2,avg=2.638992,stddev=2.475372,min=0.888640,max=4.389344,sum=5.277984
<[process 2]>[comm_wait] n=2,avg=150.230682,stddev=101.626656,min=78.369789,max=222.091583,sum=300.461365
<[process 2]>[comp_time] n=2,avg=79.961952,stddev=31.951452,min=57.368862,max=102.555038,sum=159.923904
<[process 2]>[data_proc_A] n=2,avg=5.289152,stddev=4.141677,min=2.360544,max=8.217760,sum=10.578304
<[process 2]>[data_proc_B] n=2,avg=16.816017,stddev=19.943239,min=2.714016,max=30.918016,sum=33.632034
<[process 2]>[comm_wait] n=2,avg=145.709488,stddev=56.238705,min=105.942719,max=185.476257,sum=291.418976
<[process 2]>[comp_time] n=2,avg=87.640038,stddev=80.164101,min=30.955456,max=144.324615,sum=175.280075
<[process 2]>[data_proc_A] n=2,avg=5.978832,stddev=6.271302,min=1.544352,max=10.413312,sum=11.957664
<[process 2]>[data_proc_B] n=2,avg=0.933920,stddev=0.172195,min=0.812160,max=1.055680,sum=1.867840
<[process 2]>[comm_wait] n=2,avg=104.040627,stddev=45.749763,min=71.690659,max=136.390594,sum=208.081253
<[process 2]>[comp_time] n=2,avg=82.060593,stddev=42.591110,min=51.944126,max=112.177055,sum=164.121185
<[process 2]>[data_proc_A] n=2,avg=12.535104,stddev=4.800859,min=9.140384,max=15.929824,sum=25.070208
<[process 2]>[data_proc_B] n=2,avg=5.770848,stddev=7.106186,min=0.746016,max=10.795680,sum=11.541697
<[process 2]>[comm_wait] n=2,avg=120.676834,stddev=45.082272,min=88.798851,max=152.554810,sum=241.353668
<[process 2]>[comp_time] n=2,avg=87.685341,stddev=74.598244,min=34.936417,max=140.434265,sum=175.370682
<[process 2]>[data_proc_A] n=2,avg=8.014896,stddev=0.602545,min=7.588832,max=8.440960,sum=16.029793
<[process 2]>[data_proc_B] n=2,avg=4.859728,stddev=1.447408,min=3.836256,max=5.883200,sum=9.719456
<[process 2]>[comm_wait] n=2,avg=133.942413,stddev=20.054293,min=119.761887,max=148.122940,sum=267.884827
<[process 2]>[comp_time] n=2,avg=67.772049,stddev=51.579220,min=31.300032,max=104.244064,sum=135.544098
<[process 2]>[data_proc_A] n=2,avg=5.830768,stddev=6.425123,min=1.287520,max=10.374016,sum=11.661536
<[process 2]>[data_proc_B] n=2,avg=0.839376,stddev=0.190681,min=0.704544,max=0.974208,sum=1.678752
<[process 2]>[comm_wait] n=2,avg=128.326523,stddev=32.836269,min=105.107773,max=151.545273,sum=256.653046
<[process 2]>[comp_time] n=2,avg=79.653473,stddev=52.235619,min=42.717312,max=116.589630,sum=159.306946
<[process 2]>[data_proc_A] n=2,avg=13.519888,stddev=3.815820,min=10.821696,max=16.218081,sum=27.039776
<[process 2]>[data_proc_B] n=2,avg=2.860336,stddev=2.745588,min=0.918912,max=4.801760,sum=5.720672
<[process 2]>[comm_wait] n=2,avg=172.968033,stddev=39.032154,min=145.368134,max=200.567932,sum=345.936066
<[process 2]>[comp_time] n=2,avg=84.279022,stddev=75.720047,min=30.736864,max=137.821182,sum=168.558044
<[process 2]>[data_proc_A] n=2,avg=5.815088,stddev=6.327826,min=1.340640,max=10.289536,sum=11.630177
<[process 2]>[data_proc_B] n=2,avg=0.999936,stddev=0.171561,min=0.878624,max=1.121248,sum=1.999872
<[process 2]>[comm_wait] n=2,avg=145.719498,stddev=73.290810,min=93.895073,max=197.543930,sum=291.438995
<[process 2]>[comp_time] n=2,avg=101.089806,stddev=55.482578,min=61.857697,max=140.321915,sum=202.179611
<[process 2]>[data_proc_A] n=2,avg=8.003584,stddev=0.157396,min=7.892288,max=8.114880,sum=16.007168
<[process 2]>[data_proc_B] n=2,avg=8.241505,stddev=3.162136,min=6.005536,max=10.477472,sum=16.483009
<[process 3]>[comm_wait] n=2,avg=585.637512,stddev=781.165100,min=33.270401,max=1138.004639,sum=1171.275024
<[process 3]>[comp_time] n=2,avg=426.691925,stddev=564.202026,min=27.740864,max=825.643005,sum=853.383850
<[process 3]>[data_proc_A] n=2,avg=7.780336,stddev=1.649154,min=6.614208,max=8.946464,sum=15.560672
<[process 3]>[data_proc_B] n=2,avg=2.103168,stddev=0.705115,min=1.604576,max=2.601760,sum=4.206336
<[process 3]>[comm_wait] n=2,avg=63.821617,stddev=43.930294,min=32.758209,max=94.885025,sum=127.643234
<[process 3]>[comp_time] n=2,avg=421.358856,stddev=581.848206,min=9.930048,max=832.787659,sum=842.717712
<[process 3]>[data_proc_A] n=2,avg=7.049136,stddev=2.991231,min=4.934016,max=9.164256,sum=14.098272
<[process 3]>[data_proc_B] n=2,avg=3.517952,stddev=4.074519,min=0.636832,max=6.399072,sum=7.035904
<[process 3]>[comm_wait] n=2,avg=113.556396,stddev=62.451111,min=69.396797,max=157.716003,sum=227.112793
<[process 3]>[comp_time] n=2,avg=430.533447,stddev=591.643188,min=12.178528,max=848.888367,sum=861.066895
<[process 3]>[data_proc_A] n=2,avg=10.447952,stddev=3.081877,min=8.268736,max=12.627168,sum=20.895905
<[process 3]>[data_proc_B] n=2,avg=0.889056,stddev=0.335248,min=0.652000,max=1.126112,sum=1.778112
<[process 3]>[comm_wait] n=2,avg=86.700882,stddev=60.466457,min=43.944641,max=129.457123,sum=173.401764
<[process 3]>[comp_time] n=2,avg=431.546814,stddev=584.097839,min=18.527296,max=844.566345,sum=863.093628
<[process 3]>[data_proc_A] n=2,avg=9.615601,stddev=7.697779,min=4.172448,max=15.058752,sum=19.231201
<[process 3]>[data_proc_B] n=2,avg=3.861536,stddev=4.413071,min=0.741024,max=6.982048,sum=7.723072
<[process 3]>[comm_wait] n=2,avg=47.407234,stddev=20.497812,min=32.913090,max=61.901375,sum=94.814468
<[process 3]>[comp_time] n=2,avg=416.877258,stddev=575.217957,min=10.136736,max=823.617798,sum=833.754517
<[process 3]>[data_proc_A] n=2,avg=6.424944,stddev=4.829936,min=3.009664,max=9.840224,sum=12.849888
<[process 3]>[data_proc_B] n=2,avg=0.949696,stddev=0.226908,min=0.789248,max=1.110144,sum=1.899392
<[process 3]>[comm_wait] n=2,avg=121.006142,stddev=112.407761,min=41.521854,max=200.490433,sum=242.012283
<[process 3]>[comp_time] n=2,avg=423.614471,stddev=583.651184,min=10.910784,max=836.318176,sum=847.228943
<[process 3]>[data_proc_A] n=2,avg=7.393200,stddev=2.988154,min=5.280256,max=9.506144,sum=14.786400
<[process 3]>[data_proc_B] n=2,avg=0.877248,stddev=0.407610,min=0.589024,max=1.165472,sum=1.754496
<[process 3]>[comm_wait] n=2,avg=126.492485,stddev=115.940086,min=44.510464,max=208.474503,sum=252.984970
<[process 3]>[comp_time] n=2,avg=429.475769,stddev=569.921082,min=26.480703,max=832.470825,sum=858.951538
<[process 3]>[data_proc_A] n=2,avg=9.177440,stddev=1.355111,min=8.219232,max=10.135648,sum=18.354879
<[process 3]>[data_proc_B] n=2,avg=3.991264,stddev=4.066916,min=1.115520,max=6.867008,sum=7.982528
<[process 3]>[comm_wait] n=2,avg=83.539566,stddev=85.000328,min=23.435265,max=143.643875,sum=167.079132
<[process 3]>[comp_time] n=2,avg=432.313660,stddev=582.923340,min=20.124607,max=844.502686,sum=864.627319
<[process 3]>[data_proc_A] n=2,avg=8.002912,stddev=2.788829,min=6.030912,max=9.974912,sum=16.005823
<[process 3]>[data_proc_B] n=2,avg=2.893040,stddev=3.060109,min=0.729216,max=5.056864,sum=5.786080
<[process 3]>[comm_wait] n=2,avg=59.422993,stddev=37.739929,min=32.736832,max=86.109154,sum=118.845985
<[process 3]>[comp_time] n=2,avg=424.021973,stddev=585.098206,min=10.295072,max=837.748901,sum=848.043945
<[process 3]>[data_proc_A] n=2,avg=8.603776,stddev=8.295030,min=2.738304,max=14.469248,sum=17.207552
<[process 3]>[data_proc_B] n=2,avg=3.517216,stddev=4.180415,min=0.561216,max=6.473216,sum=7.034432
<[process 0]>[comm_wait] n=2,avg=13.021376,stddev=6.222947,min=8.621088,max=17.421663,sum=26.042751
<[process 0]>[comp_time] n=2,avg=284.557159,stddev=117.858772,min=201.218430,max=367.895905,sum=569.114319
<[process 0]>[data_proc_A] n=2,avg=4.723088,stddev=4.329326,min=1.661792,max=7.784384,sum=9.446176
<[process 0]>[data_proc_B] n=2,avg=0.708000,stddev=0.290219,min=0.502784,max=0.913216,sum=1.416000
NNZ C: 929023247
<Timer>[spgemm] n=10,avg=1397.115845,stddev=318.391022,min=1231.517212,max=2296.719482,sum=13971.158203
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=31.699856,stddev=27.866816,min=11.995040,max=51.404671,sum=63.399712
<[process 0]>[comp_time] n=2,avg=298.885498,stddev=99.009193,min=228.875427,max=368.895569,sum=597.770996
<[process 0]>[data_proc_A] n=2,avg=4.872464,stddev=4.770923,min=1.498912,max=8.246016,sum=9.744927
<[process 0]>[data_proc_B] n=2,avg=0.739360,stddev=0.431731,min=0.434080,max=1.044640,sum=1.478720
NNZ C: 929023247
<Timer>[spgemm] n=11,avg=1389.244263,stddev=303.178345,min=1231.517212,max=2296.719482,sum=15281.687500
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=15.411264,stddev=6.144113,min=11.066720,max=19.755808,sum=30.822529
<[process 0]>[comp_time] n=2,avg=281.554077,stddev=123.727158,min=194.065765,max=369.042389,sum=563.108154
<[process 0]>[data_proc_A] n=2,avg=5.592464,stddev=5.599358,min=1.633120,max=9.551808,sum=11.184928
<[process 0]>[data_proc_B] n=2,avg=0.737728,stddev=0.406569,min=0.450240,max=1.025216,sum=1.475456
NNZ C: 929023247
<Timer>[spgemm] n=12,avg=1384.946411,stddev=289.452393,min=1231.517212,max=2296.719482,sum=16619.357422
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=38.755871,stddev=27.302467,min=19.450111,max=58.061630,sum=77.511742
<[process 0]>[comp_time] n=2,avg=289.366760,stddev=122.983391,min=202.404358,max=376.329132,sum=578.733521
<[process 0]>[data_proc_A] n=2,avg=7.980752,stddev=0.649430,min=7.521536,max=8.439968,sum=15.961504
<[process 0]>[data_proc_B] n=2,avg=2.712480,stddev=2.468018,min=0.967328,max=4.457632,sum=5.424960
NNZ C: 929023247
<Timer>[spgemm] n=13,avg=1377.513916,stddev=278.422241,min=1231.517212,max=2296.719482,sum=17907.681641
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=25.344448,stddev=12.174455,min=16.735807,max=33.953087,sum=50.688896
<[process 0]>[comp_time] n=2,avg=290.278625,stddev=110.400047,min=212.213989,max=368.343231,sum=580.557251
<[process 0]>[data_proc_A] n=2,avg=4.945168,stddev=4.668195,min=1.644256,max=8.246080,sum=9.890336
<[process 0]>[data_proc_B] n=2,avg=0.872976,stddev=0.524300,min=0.502240,max=1.243712,sum=1.745952
NNZ C: 929023247
<Timer>[spgemm] n=14,avg=1372.204224,stddev=268.236176,min=1231.517212,max=2296.719482,sum=19210.859375
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=8.446976,stddev=11.819476,min=0.089344,max=16.804607,sum=16.893951
<[process 0]>[comp_time] n=2,avg=302.149994,stddev=95.436226,min=234.666397,max=369.633606,sum=604.299988
<[process 0]>[data_proc_A] n=2,avg=4.169360,stddev=3.186551,min=1.916128,max=6.422592,sum=8.338720
<[process 0]>[data_proc_B] n=2,avg=2.080032,stddev=2.138834,min=0.567648,max=3.592416,sum=4.160064
NNZ C: 929023247
<Timer>[spgemm] n=15,avg=1368.611450,stddev=258.853119,min=1231.517212,max=2296.719482,sum=20529.171875
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=14.782256,stddev=2.574298,min=12.961952,max=16.602560,sum=29.564512
<[process 0]>[comp_time] n=2,avg=282.116699,stddev=120.679573,min=196.783356,max=367.450043,sum=564.233398
<[process 0]>[data_proc_A] n=2,avg=10.579345,stddev=12.456733,min=1.771104,max=19.387585,sum=21.158689
<[process 0]>[data_proc_B] n=2,avg=6.953104,stddev=9.159781,min=0.476160,max=13.430048,sum=13.906208
NNZ C: 929023247
<Timer>[spgemm] n=16,avg=1365.745117,stddev=250.338547,min=1231.517212,max=2296.719482,sum=21851.921875
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=33.796688,stddev=14.669919,min=23.423489,max=44.169888,sum=67.593376
<[process 0]>[comp_time] n=2,avg=334.151520,stddev=147.079468,min=230.150620,max=438.152405,sum=668.303040
<[process 0]>[data_proc_A] n=2,avg=10.408224,stddev=11.514460,min=2.266272,max=18.550177,sum=20.816448
<[process 0]>[data_proc_B] n=2,avg=2.715792,stddev=3.202164,min=0.451520,max=4.980064,sum=5.431584
NNZ C: 929023247
<Timer>[spgemm] n=17,avg=1363.286011,stddev=242.601227,min=1231.517212,max=2296.719482,sum=23175.861328
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=36.744896,stddev=27.312199,min=17.432257,max=56.057537,sum=73.489792
<[process 0]>[comp_time] n=2,avg=274.317871,stddev=134.250519,min=179.388412,max=369.247314,sum=548.635742
<[process 0]>[data_proc_A] n=2,avg=4.268368,stddev=3.733456,min=1.628416,max=6.908320,sum=8.536736
<[process 0]>[data_proc_B] n=2,avg=2.431744,stddev=2.435570,min=0.709536,max=4.153952,sum=4.863488
NNZ C: 929023247
<Timer>[spgemm] n=18,avg=1359.364624,stddev=235.945038,min=1231.517212,max=2296.719482,sum=24468.562500
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 1]>[comm_wait] n=2,avg=205.101440,stddev=17.039530,min=193.052673,max=217.150208,sum=410.202881
<[process 1]>[comp_time] n=2,avg=66.831940,stddev=52.776188,min=29.513536,max=104.150337,sum=133.663879
<[process 1]>[data_proc_A] n=2,avg=5.439744,stddev=1.495808,min=4.382048,max=6.497440,sum=10.879488
<[process 1]>[data_proc_B] n=2,avg=1.076336,stddev=0.263451,min=0.890048,max=1.262624,sum=2.152672
<[process 1]>[comm_wait] n=2,avg=177.711823,stddev=25.064548,min=159.988510,max=195.435135,sum=355.423645
<[process 1]>[comp_time] n=2,avg=82.564575,stddev=45.251217,min=50.567135,max=114.562019,sum=165.129150
<[process 1]>[data_proc_A] n=2,avg=6.678672,stddev=1.875971,min=5.352160,max=8.005184,sum=13.357344
<[process 1]>[data_proc_B] n=2,avg=6.146624,stddev=1.672347,min=4.964096,max=7.329152,sum=12.293248
<[process 1]>[comm_wait] n=2,avg=145.351028,stddev=30.717783,min=123.630272,max=167.071777,sum=290.702057
<[process 1]>[comp_time] n=2,avg=91.356148,stddev=83.398216,min=32.384705,max=150.327591,sum=182.712296
<[process 1]>[data_proc_A] n=2,avg=4.414000,stddev=1.455916,min=3.384512,max=5.443488,sum=8.828000
<[process 1]>[data_proc_B] n=2,avg=1.731760,stddev=0.762476,min=1.192608,max=2.270912,sum=3.463520
<[process 1]>[comm_wait] n=2,avg=117.848732,stddev=44.278278,min=86.539261,max=149.158203,sum=235.697464
<[process 1]>[comp_time] n=2,avg=70.734108,stddev=58.709591,min=29.220160,max=112.248062,sum=141.468216
<[process 1]>[data_proc_A] n=2,avg=8.770272,stddev=8.167321,min=2.995104,max=14.545440,sum=17.540545
<[process 1]>[data_proc_B] n=2,avg=2.809392,stddev=2.151800,min=1.287840,max=4.330944,sum=5.618784
<[process 1]>[comm_wait] n=2,avg=163.776382,stddev=77.144554,min=109.226944,max=218.325821,sum=327.552765
<[process 1]>[comp_time] n=2,avg=66.948547,stddev=52.695091,min=29.687489,max=104.209602,sum=133.897095
<[process 1]>[data_proc_A] n=2,avg=4.930448,stddev=0.979201,min=4.238048,max=5.622848,sum=9.860896
<[process 1]>[data_proc_B] n=2,avg=0.872304,stddev=0.424739,min=0.571968,max=1.172640,sum=1.744608
<[process 1]>[comm_wait] n=2,avg=188.078735,stddev=1.564620,min=186.972382,max=189.185089,sum=376.157471
<[process 1]>[comp_time] n=2,avg=76.946625,stddev=50.110409,min=41.513214,max=112.380035,sum=153.893250
<[process 1]>[data_proc_A] n=2,avg=8.723969,stddev=4.409857,min=5.605728,max=11.842208,sum=17.447937
<[process 1]>[data_proc_B] n=2,avg=3.460576,stddev=3.887571,min=0.711648,max=6.209504,sum=6.921152
<[process 1]>[comm_wait] n=2,avg=200.311768,stddev=12.664187,min=191.356827,max=209.266693,sum=400.623535
<[process 1]>[comp_time] n=2,avg=91.410309,stddev=84.876663,min=31.393440,max=151.427170,sum=182.820618
<[process 1]>[data_proc_A] n=2,avg=5.467440,stddev=1.340516,min=4.519552,max=6.415328,sum=10.934880
<[process 1]>[data_proc_B] n=2,avg=0.975984,stddev=0.403605,min=0.690592,max=1.261376,sum=1.951968
<[process 1]>[comm_wait] n=2,avg=193.184738,stddev=3.168105,min=190.944550,max=195.424927,sum=386.369476
<[process 1]>[comp_time] n=2,avg=89.443573,stddev=83.695267,min=30.262079,max=148.625061,sum=178.887146
<[process 1]>[data_proc_A] n=2,avg=3.654528,stddev=2.712982,min=1.736160,max=5.572896,sum=7.309056
<[process 1]>[data_proc_B] n=2,avg=1.291296,stddev=0.223785,min=1.133056,max=1.449536,sum=2.582592
<[process 1]>[comm_wait] n=2,avg=162.852264,stddev=7.851943,min=157.300095,max=168.404419,sum=325.704529
<[process 1]>[comp_time] n=2,avg=67.373680,stddev=50.685165,min=31.533855,max=103.213501,sum=134.747360
<[process 1]>[data_proc_A] n=2,avg=4.697472,stddev=1.979990,min=3.297408,max=6.097536,sum=9.394944
<[process 1]>[data_proc_B] n=2,avg=0.924080,stddev=0.374914,min=0.658976,max=1.189184,sum=1.848160
<[process 1]>[comm_wait] n=2,avg=178.938599,stddev=18.205744,min=166.065186,max=191.811996,sum=357.877197
<[process 2]>[comm_wait] n=2,avg=134.806152,stddev=11.575194,min=126.621246,max=142.991043,sum=269.612305
<[process 2]>[comp_time] n=2,avg=68.658302,stddev=53.189545,min=31.047615,max=106.268990,sum=137.316605
<[process 2]>[data_proc_A] n=2,avg=7.749887,stddev=1.852687,min=6.439840,max=9.059936,sum=15.499775
<[process 2]>[data_proc_B] n=2,avg=0.855792,stddev=0.235846,min=0.689024,max=1.022560,sum=1.711584
<[process 2]>[comm_wait] n=2,avg=156.766357,stddev=67.357368,min=109.137505,max=204.395203,sum=313.532715
<[process 2]>[comp_time] n=2,avg=107.824989,stddev=8.569592,min=101.765373,max=113.884605,sum=215.649979
<[process 2]>[data_proc_A] n=2,avg=10.240544,stddev=3.900423,min=7.482528,max=12.998560,sum=20.481089
<[process 2]>[data_proc_B] n=2,avg=2.195520,stddev=1.712488,min=0.984608,max=3.406432,sum=4.391040
<[process 2]>[comm_wait] n=2,avg=146.797348,stddev=68.015572,min=98.703072,max=194.891617,sum=293.594696
<[process 2]>[comp_time] n=2,avg=101.367554,stddev=69.105812,min=52.502369,max=150.232742,sum=202.735107
<[process 2]>[data_proc_A] n=2,avg=14.111232,stddev=15.683877,min=3.021056,max=25.201408,sum=28.222464
<[process 2]>[data_proc_B] n=2,avg=11.156288,stddev=14.258758,min=1.073824,max=21.238752,sum=22.312576
<[process 2]>[comm_wait] n=2,avg=175.401749,stddev=104.759529,min=101.325569,max=249.477921,sum=350.803497
<[process 2]>[comp_time] n=2,avg=70.465652,stddev=49.734409,min=35.298111,max=105.633186,sum=140.931305
<[process 2]>[data_proc_A] n=2,avg=9.231488,stddev=0.781778,min=8.678688,max=9.784288,sum=18.462976
<[process 2]>[data_proc_B] n=2,avg=4.501840,stddev=0.930733,min=3.843712,max=5.159968,sum=9.003680
<[process 2]>[comm_wait] n=2,avg=129.689072,stddev=55.573822,min=90.392448,max=168.985703,sum=259.378143
<[process 2]>[comp_time] n=2,avg=67.870590,stddev=51.527653,min=31.435040,max=104.306145,sum=135.741180
<[process 2]>[data_proc_A] n=2,avg=13.105152,stddev=5.597887,min=9.146848,max=17.063456,sum=26.210304
<[process 2]>[data_proc_B] n=2,avg=1.171840,stddev=0.694435,min=0.680800,max=1.662880,sum=2.343680
<[process 2]>[comm_wait] n=2,avg=135.565811,stddev=6.328463,min=131.090912,max=140.040710,sum=271.131622
<[process 2]>[comp_time] n=2,avg=80.720192,stddev=53.983311,min=42.548225,max=118.892159,sum=161.440384
<[process 2]>[data_proc_A] n=2,avg=8.899263,stddev=10.402050,min=1.543904,max=16.254623,sum=17.798527
<[process 2]>[data_proc_B] n=2,avg=3.174256,stddev=2.771836,min=1.214272,max=5.134240,sum=6.348512
<[process 2]>[comm_wait] n=2,avg=167.203857,stddev=51.150784,min=131.034790,max=203.372925,sum=334.407715
<[process 2]>[comp_time] n=2,avg=90.705055,stddev=83.563103,min=31.617023,max=149.793091,sum=181.410110
<[process 2]>[data_proc_A] n=2,avg=5.395248,stddev=5.630131,min=1.414144,max=9.376352,sum=10.790497
<[process 2]>[data_proc_B] n=2,avg=0.924272,stddev=0.208014,min=0.777184,max=1.071360,sum=1.848544
<[process 2]>[comm_wait] n=2,avg=161.867706,stddev=74.870232,min=108.926460,max=214.808960,sum=323.735413
<[process 2]>[comp_time] n=2,avg=108.065346,stddev=60.408504,min=65.350082,max=150.780609,sum=216.130692
<[process 2]>[data_proc_A] n=2,avg=19.616400,stddev=5.217499,min=15.927072,max=23.305729,sum=39.232800
<[process 2]>[data_proc_B] n=2,avg=6.006624,stddev=7.010562,min=1.049408,max=10.963840,sum=12.013247
<[process 2]>[comm_wait] n=2,avg=174.728058,stddev=112.036301,min=95.506432,max=253.949692,sum=349.456116
<[process 2]>[comp_time] n=2,avg=67.767937,stddev=48.033978,min=33.802784,max=101.733086,sum=135.535873
<[process 2]>[data_proc_A] n=2,avg=4.807152,stddev=4.906280,min=1.337888,max=8.276416,sum=9.614304
<[process 2]>[data_proc_B] n=2,avg=2.617744,stddev=1.855697,min=1.305568,max=3.929920,sum=5.235488
<[process 2]>[comm_wait] n=2,avg=165.754425,stddev=90.272331,min=101.922241,max=229.586594,sum=331.508850
<[process 3]>[comm_wait] n=2,avg=128.136963,stddev=92.870567,min=62.467552,max=193.806366,sum=256.273926
<[process 3]>[comp_time] n=2,avg=423.323029,stddev=583.244812,min=10.906688,max=835.739380,sum=846.646057
<[process 3]>[data_proc_A] n=2,avg=7.113616,stddev=4.417211,min=3.990176,max=10.237056,sum=14.227232
<[process 3]>[data_proc_B] n=2,avg=1.370816,stddev=1.078287,min=0.608352,max=2.133280,sum=2.741632
<[process 3]>[comm_wait] n=2,avg=48.835358,stddev=2.287904,min=47.217567,max=50.453152,sum=97.670715
<[process 3]>[comp_time] n=2,avg=436.743683,stddev=559.995544,min=40.767010,max=832.720337,sum=873.487366
<[process 3]>[data_proc_A] n=2,avg=7.690544,stddev=1.304584,min=6.768064,max=8.613024,sum=15.381088
<[process 3]>[data_proc_B] n=2,avg=3.205536,stddev=2.640439,min=1.338464,max=5.072608,sum=6.411072
<[process 3]>[comm_wait] n=2,avg=68.866028,stddev=44.093075,min=37.687519,max=100.044540,sum=137.732056
<[process 3]>[comp_time] n=2,avg=430.451660,stddev=593.532776,min=10.760640,max=850.142700,sum=860.903320
<[process 3]>[data_proc_A] n=2,avg=6.520832,stddev=5.050665,min=2.949472,max=10.092192,sum=13.041664
<[process 3]>[data_proc_B] n=2,avg=0.855040,stddev=0.342489,min=0.612864,max=1.097216,sum=1.710080
<[process 3]>[comm_wait] n=2,avg=62.693375,stddev=35.121189,min=37.858944,max=87.527809,sum=125.386749
<[process 3]>[comp_time] n=2,avg=419.032654,stddev=577.873352,min=10.414496,max=827.650818,sum=838.065308
<[process 3]>[data_proc_A] n=2,avg=6.259696,stddev=4.955156,min=2.755872,max=9.763520,sum=12.519392
<[process 3]>[data_proc_B] n=2,avg=0.995248,stddev=0.348485,min=0.748832,max=1.241664,sum=1.990496
<[process 3]>[comm_wait] n=2,avg=109.717407,stddev=83.218712,min=50.872894,max=168.561920,sum=219.434814
<[process 3]>[comp_time] n=2,avg=424.082855,stddev=584.392639,min=10.854848,max=837.310852,sum=848.165710
<[process 3]>[data_proc_A] n=2,avg=6.693360,stddev=3.762239,min=4.033056,max=9.353664,sum=13.386721
<[process 3]>[data_proc_B] n=2,avg=0.911312,stddev=0.415824,min=0.617280,max=1.205344,sum=1.822624
<[process 3]>[comm_wait] n=2,avg=121.175125,stddev=98.390732,min=51.602367,max=190.747879,sum=242.350250
<[process 3]>[comp_time] n=2,avg=429.624512,stddev=561.203186,min=32.793919,max=826.455078,sum=859.249023
<[process 3]>[data_proc_A] n=2,avg=9.749584,stddev=1.874749,min=8.423936,max=11.075232,sum=19.499168
<[process 3]>[data_proc_B] n=2,avg=3.686432,stddev=3.107695,min=1.488960,max=5.883904,sum=7.372864
<[process 3]>[comm_wait] n=2,avg=139.022583,stddev=105.455124,min=64.454559,max=213.590622,sum=278.045166
<[process 3]>[comp_time] n=2,avg=427.796631,stddev=586.991272,min=12.731168,max=842.862122,sum=855.593262
<[process 3]>[data_proc_A] n=2,avg=6.575568,stddev=3.602172,min=4.028448,max=9.122688,sum=13.151136
<[process 3]>[data_proc_B] n=2,avg=0.897728,stddev=0.400867,min=0.614272,max=1.181184,sum=1.795456
<[process 3]>[comm_wait] n=2,avg=62.408688,stddev=31.210922,min=40.339233,max=84.478142,sum=124.817375
<[process 3]>[comp_time] n=2,avg=430.399109,stddev=593.228760,min=10.923008,max=849.875183,sum=860.798218
<[process 3]>[data_proc_A] n=2,avg=6.450448,stddev=7.167935,min=1.381952,max=11.518944,sum=12.900896
<[process 3]>[data_proc_B] n=2,avg=1.025872,stddev=0.639066,min=0.573984,max=1.477760,sum=2.051744
<[process 3]>[comm_wait] n=2,avg=67.006592,stddev=43.806770,min=36.030529,max=97.982658,sum=134.013184
<[process 3]>[comp_time] n=2,avg=420.459381,stddev=575.357117,min=13.620448,max=827.298340,sum=840.918762
<[process 3]>[data_proc_A] n=2,avg=6.408800,stddev=4.909109,min=2.937536,max=9.880064,sum=12.817600
<[process 3]>[data_proc_B] n=2,avg=0.906592,stddev=0.337194,min=0.668160,max=1.145024,sum=1.813184
<[process 0]>[comm_wait] n=2,avg=30.059824,stddev=19.504404,min=16.268127,max=43.851521,sum=60.119648
<[process 0]>[comp_time] n=2,avg=305.970947,stddev=89.196564,min=242.899460,max=369.042450,sum=611.941895
<[process 0]>[data_proc_A] n=2,avg=8.486720,stddev=9.660775,min=1.655520,max=15.317920,sum=16.973440
<[process 0]>[data_proc_B] n=2,avg=6.711600,stddev=8.868929,min=0.440320,max=12.982880,sum=13.423200
NNZ C: 929023247
<[process 3]>[comm_wait] n=2,avg=65.200111,stddev=36.130669,min=39.651871,max=90.748352,sum=130.400223
<Timer>[spgemm] n=19,avg=1355.969238,stddev=229.774475,min=1231.517212,max=2296.719482,sum=25763.416016
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=18.272928,stddev=0.674026,min=17.796320,max=18.749537,sum=36.545856
<[process 0]>[comp_time] n=2,avg=304.510223,stddev=182.214035,min=175.665436,max=433.355011,sum=609.020447
<[process 0]>[data_proc_A] n=2,avg=13.756528,stddev=9.678538,min=6.912768,max=20.600288,sum=27.513056
<[process 0]>[data_proc_B] n=2,avg=5.664432,stddev=1.452793,min=4.637152,max=6.691712,sum=11.328864
NNZ C: 929023247
<Timer>[spgemm] n=20,avg=1354.113770,stddev=223.799957,min=1231.517212,max=2296.719482,sum=27082.275391
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=36.122063,stddev=46.007893,min=3.589568,max=68.654556,sum=72.244125
<[process 0]>[comp_time] n=2,avg=289.379608,stddev=191.074234,min=154.269730,max=424.489502,sum=578.759216
<[process 0]>[data_proc_A] n=2,avg=7.409792,stddev=1.748014,min=6.173760,max=8.645824,sum=14.819584
<[process 0]>[data_proc_B] n=2,avg=3.860368,stddev=3.913977,min=1.092768,max=6.627968,sum=7.720736
NNZ C: 929023247
<Timer>[spgemm] n=21,avg=1351.388916,stddev=218.490295,min=1231.517212,max=2296.719482,sum=28379.167969
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=16.277584,stddev=0.899779,min=15.641344,max=16.913824,sum=32.555168
<[process 0]>[comp_time] n=2,avg=288.751801,stddev=116.822075,min=206.146118,max=371.357483,sum=577.503601
<[process 0]>[data_proc_A] n=2,avg=3.861376,stddev=3.398547,min=1.458240,max=6.264512,sum=7.722752
<[process 0]>[data_proc_B] n=2,avg=2.117312,stddev=2.365832,min=0.444416,max=3.790208,sum=4.234624
NNZ C: 929023247
<Timer>[spgemm] n=22,avg=1349.036255,stddev=213.510056,min=1231.517212,max=2296.719482,sum=29678.796875
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=21.134705,stddev=11.241957,min=13.185440,max=29.083967,sum=42.269409
<[process 0]>[comp_time] n=2,avg=301.496887,stddev=103.519836,min=228.297318,max=374.696472,sum=602.993774
<[process 0]>[data_proc_A] n=2,avg=3.916576,stddev=1.966006,min=2.526400,max=5.306752,sum=7.833152
<[process 0]>[data_proc_B] n=2,avg=1.505536,stddev=1.503275,min=0.442560,max=2.568512,sum=3.011072
NNZ C: 929023247
<Timer>[spgemm] n=23,avg=1345.303223,stddev=209.367950,min=1231.517212,max=2296.719482,sum=30941.974609
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=23.837440,stddev=10.858626,min=16.159231,max=31.515648,sum=47.674881
<[process 0]>[comp_time] n=2,avg=296.362549,stddev=104.860291,min=222.215134,max=370.509979,sum=592.725098
<[process 0]>[data_proc_A] n=2,avg=12.199471,stddev=14.439889,min=1.988928,max=22.410015,sum=24.398943
<[process 0]>[data_proc_B] n=2,avg=2.724000,stddev=3.070088,min=0.553120,max=4.894880,sum=5.448000
NNZ C: 929023247
<Timer>[spgemm] n=24,avg=1343.907349,stddev=204.880066,min=1231.517212,max=2296.719482,sum=32253.775391
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=18.335201,stddev=0.066117,min=18.288448,max=18.381952,sum=36.670403
<[process 0]>[comp_time] n=2,avg=284.815399,stddev=141.579193,min=184.703781,max=384.927002,sum=569.630798
<[process 0]>[data_proc_A] n=2,avg=4.849712,stddev=4.395670,min=1.741504,max=7.957920,sum=9.699424
<[process 0]>[data_proc_B] n=2,avg=0.824608,stddev=0.158030,min=0.712864,max=0.936352,sum=1.649216
NNZ C: 929023247
<Timer>[spgemm] n=25,avg=1343.622192,stddev=200.571396,min=1231.517212,max=2296.719482,sum=33590.554688
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=58.515648,stddev=42.053776,min=28.779137,max=88.252159,sum=117.031296
<[process 0]>[comp_time] n=2,avg=258.523895,stddev=155.796234,min=148.359329,max=368.688477,sum=517.047791
<[process 0]>[data_proc_A] n=2,avg=46.197998,stddev=63.266460,min=1.461856,max=90.934143,sum=92.395996
<[process 0]>[data_proc_B] n=2,avg=19.478224,stddev=26.909000,min=0.450688,max=38.505760,sum=38.956448
NNZ C: 929023247
<Timer>[spgemm] n=26,avg=1341.187256,stddev=196.910873,min=1231.517212,max=2296.719482,sum=34870.867188
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=24.311312,stddev=5.621985,min=20.335968,max=28.286655,sum=48.622623
<[process 0]>[comp_time] n=2,avg=295.158997,stddev=106.199196,min=220.064835,max=370.253174,sum=590.317993
<[process 0]>[data_proc_A] n=2,avg=5.540576,stddev=4.982919,min=2.017120,max=9.064032,sum=11.081152
<[process 0]>[data_proc_B] n=2,avg=0.811152,stddev=0.241276,min=0.640544,max=0.981760,sum=1.622304
NNZ C: 929023247
<Timer>[spgemm] n=27,avg=1340.546265,stddev=193.115753,min=1231.517212,max=2296.719482,sum=36194.750000
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 1]>[comp_time] n=2,avg=84.268173,stddev=40.924198,min=55.330399,max=113.205956,sum=168.536346
<[process 1]>[data_proc_A] n=2,avg=8.117184,stddev=4.226484,min=5.128608,max=11.105760,sum=16.234367
<[process 1]>[data_proc_B] n=2,avg=2.703696,stddev=3.027164,min=0.563168,max=4.844224,sum=5.407392
<[process 1]>[comm_wait] n=2,avg=183.954987,stddev=44.755676,min=152.307938,max=215.602020,sum=367.909973
<[process 1]>[comp_time] n=2,avg=86.218384,stddev=79.646896,min=29.899521,max=142.537247,sum=172.436768
<[process 1]>[data_proc_A] n=2,avg=3.706272,stddev=2.052850,min=2.254688,max=5.157856,sum=7.412544
<[process 1]>[data_proc_B] n=2,avg=0.886224,stddev=0.483842,min=0.544096,max=1.228352,sum=1.772448
<[process 1]>[comm_wait] n=2,avg=171.480438,stddev=22.846004,min=155.325882,max=187.635010,sum=342.960876
<[process 1]>[comp_time] n=2,avg=80.221695,stddev=71.655418,min=29.553663,max=130.889725,sum=160.443390
<[process 1]>[data_proc_A] n=2,avg=8.107648,stddev=5.462258,min=4.245248,max=11.970048,sum=16.215296
<[process 1]>[data_proc_B] n=2,avg=3.053968,stddev=2.547463,min=1.252640,max=4.855296,sum=6.107936
<[process 1]>[comm_wait] n=2,avg=128.275146,stddev=100.820152,min=56.984543,max=199.565765,sum=256.550293
<[process 1]>[comp_time] n=2,avg=67.442513,stddev=50.718426,min=31.579168,max=103.305855,sum=134.885025
<[process 1]>[data_proc_A] n=2,avg=4.631856,stddev=2.261633,min=3.032640,max=6.231072,sum=9.263712
<[process 1]>[data_proc_B] n=2,avg=0.902368,stddev=0.444855,min=0.587808,max=1.216928,sum=1.804736
<[process 1]>[comm_wait] n=2,avg=182.664703,stddev=80.950584,min=125.424004,max=239.905411,sum=365.329407
<[process 1]>[comp_time] n=2,avg=67.471565,stddev=51.393764,min=31.130688,max=103.812447,sum=134.943130
<[process 1]>[data_proc_A] n=2,avg=4.359104,stddev=1.566632,min=3.251328,max=5.466880,sum=8.718208
<[process 1]>[data_proc_B] n=2,avg=0.988592,stddev=0.421255,min=0.690720,max=1.286464,sum=1.977184
<[process 1]>[comm_wait] n=2,avg=184.179047,stddev=36.371815,min=158.460281,max=209.897797,sum=368.358093
<[process 1]>[comp_time] n=2,avg=66.963501,stddev=50.269184,min=31.417824,max=102.509186,sum=133.927002
<[process 1]>[data_proc_A] n=2,avg=10.917680,stddev=3.055402,min=8.757184,max=13.078176,sum=21.835360
<[process 1]>[data_proc_B] n=2,avg=2.371008,stddev=2.513544,min=0.593664,max=4.148352,sum=4.742016
<[process 1]>[comm_wait] n=2,avg=180.632355,stddev=27.469593,min=161.208420,max=200.056290,sum=361.264709
<[process 1]>[comp_time] n=2,avg=88.200653,stddev=82.813469,min=29.642689,max=146.758621,sum=176.401306
<[process 1]>[data_proc_A] n=2,avg=5.255632,stddev=1.179816,min=4.421376,max=6.089888,sum=10.511265
<[process 1]>[data_proc_B] n=2,avg=1.047344,stddev=0.144748,min=0.944992,max=1.149696,sum=2.094688
<[process 1]>[comm_wait] n=2,avg=112.561699,stddev=104.056206,min=38.982849,max=186.140549,sum=225.123398
<[process 1]>[comp_time] n=2,avg=71.867584,stddev=45.261082,min=39.863167,max=103.872002,sum=143.735168
<[process 1]>[data_proc_A] n=2,avg=7.601328,stddev=2.766225,min=5.645312,max=9.557344,sum=15.202656
<[process 1]>[data_proc_B] n=2,avg=3.285104,stddev=3.829532,min=0.577216,max=5.992992,sum=6.570208
<[process 1]>[comm_wait] n=2,avg=153.649185,stddev=30.937927,min=131.772766,max=175.525604,sum=307.298370
<[process 1]>[comp_time] n=2,avg=91.261139,stddev=86.940216,min=29.785120,max=152.737152,sum=182.522278
<[process 1]>[data_proc_A] n=2,avg=4.284800,stddev=1.734437,min=3.058368,max=5.511232,sum=8.569600
<[process 1]>[data_proc_B] n=2,avg=0.996192,stddev=0.256595,min=0.814752,max=1.177632,sum=1.992384
<[process 1]>[comm_wait] n=2,avg=139.973724,stddev=79.784813,min=83.557343,max=196.390106,sum=279.947449
<[process 1]>[comp_time] n=2,avg=66.988892,stddev=50.202362,min=31.490463,max=102.487328,sum=133.977783
<[process 2]>[comp_time] n=2,avg=115.073090,stddev=6.403694,min=110.544991,max=119.601181,sum=230.146179
<[process 2]>[data_proc_A] n=2,avg=12.328656,stddev=7.286051,min=7.176640,max=17.480673,sum=24.657312
<[process 2]>[data_proc_B] n=2,avg=3.090784,stddev=2.890653,min=1.046784,max=5.134784,sum=6.181568
<[process 2]>[comm_wait] n=2,avg=155.740814,stddev=38.196907,min=128.731522,max=182.750107,sum=311.481628
<[process 2]>[comp_time] n=2,avg=87.823761,stddev=80.061798,min=31.211519,max=144.436005,sum=175.647522
<[process 2]>[data_proc_A] n=2,avg=8.294800,stddev=1.806233,min=7.017600,max=9.572000,sum=16.589600
<[process 2]>[data_proc_B] n=2,avg=3.077552,stddev=3.411151,min=0.665504,max=5.489600,sum=6.155104
<[process 2]>[comm_wait] n=2,avg=105.093445,stddev=8.764230,min=98.896194,max=111.290688,sum=210.186890
<[process 2]>[comp_time] n=2,avg=83.080078,stddev=73.076164,min=31.407425,max=134.752731,sum=166.160156
<[process 2]>[data_proc_A] n=2,avg=8.778096,stddev=0.278929,min=8.580864,max=8.975328,sum=17.556192
<[process 2]>[data_proc_B] n=2,avg=3.870784,stddev=4.527973,min=0.669024,max=7.072544,sum=7.741568
<[process 2]>[comm_wait] n=2,avg=135.185074,stddev=43.714294,min=104.274399,max=166.095749,sum=270.370148
<[process 2]>[comp_time] n=2,avg=89.311554,stddev=43.393185,min=58.627937,max=119.995171,sum=178.623108
<[process 2]>[data_proc_A] n=2,avg=4.587888,stddev=4.658419,min=1.293888,max=7.881888,sum=9.175776
<[process 2]>[data_proc_B] n=2,avg=3.479280,stddev=1.327754,min=2.540416,max=4.418144,sum=6.958560
<[process 2]>[comm_wait] n=2,avg=155.080170,stddev=38.148712,min=128.104965,max=182.055389,sum=310.160339
<[process 2]>[comp_time] n=2,avg=70.982964,stddev=51.258957,min=34.737408,max=107.228516,sum=141.965927
<[process 2]>[data_proc_A] n=2,avg=19.040207,stddev=15.662358,min=7.965248,max=30.115168,sum=38.080414
<[process 2]>[data_proc_B] n=2,avg=0.905232,stddev=0.269922,min=0.714368,max=1.096096,sum=1.810464
<[process 2]>[comm_wait] n=2,avg=147.750641,stddev=78.884270,min=91.971039,max=203.530243,sum=295.501282
<[process 2]>[comp_time] n=2,avg=82.227295,stddev=39.017765,min=54.637569,max=109.817024,sum=164.454590
<[process 2]>[data_proc_A] n=2,avg=8.629552,stddev=1.859815,min=7.314464,max=9.944640,sum=17.259104
<[process 2]>[data_proc_B] n=2,avg=13.861616,stddev=12.949557,min=4.704896,max=23.018335,sum=27.723232
<[process 2]>[comm_wait] n=2,avg=122.102112,stddev=32.069843,min=99.425308,max=144.778915,sum=244.204224
<[process 2]>[comp_time] n=2,avg=85.251915,stddev=76.662659,min=31.043232,max=139.460602,sum=170.503830
<[process 2]>[data_proc_A] n=2,avg=10.295551,stddev=1.607361,min=9.158976,max=11.432128,sum=20.591103
<[process 2]>[data_proc_B] n=2,avg=0.962800,stddev=0.433881,min=0.656000,max=1.269600,sum=1.925600
<[process 2]>[comm_wait] n=2,avg=77.620705,stddev=11.616098,min=69.406883,max=85.834526,sum=155.241409
<[process 2]>[comp_time] n=2,avg=74.863838,stddev=44.584743,min=43.337666,max=106.390015,sum=149.727676
<[process 2]>[data_proc_A] n=2,avg=4.810192,stddev=4.523379,min=1.611680,max=8.008704,sum=9.620384
<[process 2]>[data_proc_B] n=2,avg=2.361408,stddev=1.591070,min=1.236352,max=3.486464,sum=4.722816
<[process 2]>[comm_wait] n=2,avg=143.435165,stddev=71.543182,min=92.846497,max=194.023834,sum=286.870331
<[process 2]>[comp_time] n=2,avg=91.506111,stddev=76.671509,min=37.291168,max=145.721054,sum=183.012222
<[process 2]>[data_proc_A] n=2,avg=20.324959,stddev=3.172997,min=18.081312,max=22.568607,sum=40.649918
<[process 2]>[data_proc_B] n=2,avg=7.483312,stddev=0.854886,min=6.878816,max=8.087808,sum=14.966623
<[process 2]>[comm_wait] n=2,avg=161.694702,stddev=84.920227,min=101.647041,max=221.742371,sum=323.389404
<[process 2]>[comp_time] n=2,avg=71.430977,stddev=50.105698,min=36.000896,max=106.861053,sum=142.861954
<[process 3]>[comp_time] n=2,avg=435.105713,stddev=553.864929,min=43.464066,max=826.747375,sum=870.211426
<[process 3]>[data_proc_A] n=2,avg=10.186815,stddev=0.963701,min=9.505376,max=10.868256,sum=20.373631
<[process 3]>[data_proc_B] n=2,avg=2.813056,stddev=2.185673,min=1.267552,max=4.358560,sum=5.626112
<[process 3]>[comm_wait] n=2,avg=74.464523,stddev=24.655035,min=57.030785,max=91.898270,sum=148.929047
<[process 3]>[comp_time] n=2,avg=430.287933,stddev=593.042236,min=10.943712,max=849.632141,sum=860.575867
<[process 3]>[data_proc_A] n=2,avg=7.034016,stddev=7.299243,min=1.872672,max=12.195360,sum=14.068032
<[process 3]>[data_proc_B] n=2,avg=1.120992,stddev=0.692218,min=0.631520,max=1.610464,sum=2.241984
<[process 3]>[comm_wait] n=2,avg=104.629547,stddev=118.436172,min=20.882528,max=188.376572,sum=209.259094
<[process 3]>[comp_time] n=2,avg=428.536072,stddev=590.611572,min=10.910656,max=846.161499,sum=857.072144
<[process 3]>[data_proc_A] n=2,avg=9.275616,stddev=7.705315,min=3.827136,max=14.724096,sum=18.551231
<[process 3]>[data_proc_B] n=2,avg=2.685120,stddev=2.962744,min=0.590144,max=4.780096,sum=5.370240
<[process 3]>[comm_wait] n=2,avg=72.829536,stddev=37.223820,min=46.508320,max=99.150749,sum=145.659073
<[process 3]>[comp_time] n=2,avg=418.799255,stddev=570.742126,min=15.223616,max=822.374878,sum=837.598511
<[process 3]>[data_proc_A] n=2,avg=6.086336,stddev=4.818735,min=2.678976,max=9.493696,sum=12.172672
<[process 3]>[data_proc_B] n=2,avg=0.901264,stddev=0.290559,min=0.695808,max=1.106720,sum=1.802528
<[process 3]>[comm_wait] n=2,avg=34.215023,stddev=1.970056,min=32.821983,max=35.608063,sum=68.430046
<[process 3]>[comp_time] n=2,avg=426.127167,stddev=570.062561,min=23.032032,max=829.222290,sum=852.254333
<[process 3]>[data_proc_A] n=2,avg=6.188128,stddev=4.965405,min=2.677056,max=9.699200,sum=12.376256
<[process 3]>[data_proc_B] n=2,avg=1.005536,stddev=0.419422,min=0.708960,max=1.302112,sum=2.011072
<[process 3]>[comm_wait] n=2,avg=79.156670,stddev=55.722099,min=39.755199,max=118.558144,sum=158.313339
<[process 3]>[comp_time] n=2,avg=426.111176,stddev=588.576355,min=9.924832,max=842.297546,sum=852.222351
<[process 3]>[data_proc_A] n=2,avg=11.282127,stddev=3.253121,min=8.981824,max=13.582432,sum=22.564255
<[process 3]>[data_proc_B] n=2,avg=2.132704,stddev=1.304833,min=1.210048,max=3.055360,sum=4.265408
<[process 3]>[comm_wait] n=2,avg=118.922440,stddev=116.960533,min=36.218849,max=201.626022,sum=237.844879
<[process 3]>[comp_time] n=2,avg=430.711945,stddev=593.819153,min=10.818432,max=850.605469,sum=861.423889
<[process 3]>[data_proc_A] n=2,avg=7.585088,stddev=4.992694,min=4.054720,max=11.115456,sum=15.170176
<[process 3]>[data_proc_B] n=2,avg=3.640912,stddev=4.225331,min=0.653152,max=6.628672,sum=7.281824
<[process 3]>[comm_wait] n=2,avg=67.484032,stddev=49.885578,min=32.209599,max=102.758461,sum=134.968063
<[process 3]>[comp_time] n=2,avg=426.611542,stddev=573.180847,min=21.311487,max=831.911621,sum=853.223083
<[process 3]>[data_proc_A] n=2,avg=8.589920,stddev=0.465491,min=8.260768,max=8.919072,sum=17.179840
<[process 3]>[data_proc_B] n=2,avg=3.350624,stddev=3.179107,min=1.102656,max=5.598592,sum=6.701248
<[process 3]>[comm_wait] n=2,avg=72.464478,stddev=51.004463,min=36.398880,max=108.530083,sum=144.928955
<[process 3]>[comp_time] n=2,avg=431.735992,stddev=596.073181,min=10.248608,max=853.223389,sum=863.471985
<[process 3]>[data_proc_A] n=2,avg=6.270128,stddev=5.025708,min=2.716416,max=9.823840,sum=12.540257
<[process 3]>[data_proc_B] n=2,avg=1.098368,stddev=0.562744,min=0.700448,max=1.496288,sum=2.196736
<[process 3]>[comm_wait] n=2,avg=68.186829,stddev=34.912315,min=43.500095,max=92.873566,sum=136.373657
<[process 3]>[comp_time] n=2,avg=419.731750,stddev=573.993042,min=13.857376,max=825.606140,sum=839.463501
<[process 0]>[comm_wait] n=2,avg=20.386496,stddev=8.521711,min=14.360736,max=26.412256,sum=40.772991
<[process 0]>[comp_time] n=2,avg=296.977600,stddev=111.216866,min=218.335388,max=375.619781,sum=593.955200
<[process 0]>[data_proc_A] n=2,avg=10.500688,stddev=12.609512,min=1.584416,max=19.416960,sum=21.001375
<[process 0]>[data_proc_B] n=2,avg=0.752672,stddev=0.389554,min=0.477216,max=1.028128,sum=1.505344
NNZ C: 929023247
<Timer>[spgemm] n=28,avg=1338.810425,stddev=189.728226,min=1231.517212,max=2296.719482,sum=37486.691406
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=23.145840,stddev=9.705012,min=16.283360,max=30.008320,sum=46.291679
<[process 0]>[comp_time] n=2,avg=294.065948,stddev=106.977554,min=218.421402,max=369.710510,sum=588.131897
<[process 0]>[data_proc_A] n=2,avg=4.551056,stddev=3.271585,min=2.237696,max=6.864416,sum=9.102112
<[process 0]>[data_proc_B] n=2,avg=0.788272,stddev=0.279562,min=0.590592,max=0.985952,sum=1.576544
NNZ C: 929023247
<Timer>[spgemm] n=29,avg=1337.797607,stddev=186.389267,min=1231.517212,max=2296.719482,sum=38796.128906
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=23.705008,stddev=8.875220,min=17.429279,max=29.980736,sum=47.410015
<[process 0]>[comp_time] n=2,avg=275.497986,stddev=130.309204,min=183.355453,max=367.640503,sum=550.995972
<[process 0]>[data_proc_A] n=2,avg=45.025661,stddev=61.456654,min=1.569248,max=88.482079,sum=90.051323
<[process 0]>[data_proc_B] n=2,avg=2.639712,stddev=3.103848,min=0.444960,max=4.834464,sum=5.279424
NNZ C: 929023247
<Timer>[spgemm] n=30,avg=1335.815796,stddev=183.468826,min=1231.517212,max=2296.719482,sum=40074.472656
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=32.864944,stddev=8.268714,min=27.018080,max=38.711807,sum=65.729889
<[process 0]>[comp_time] n=2,avg=303.256836,stddev=99.064606,min=233.207581,max=373.306091,sum=606.513672
<[process 0]>[data_proc_A] n=2,avg=10.283919,stddev=11.635854,min=2.056128,max=18.511711,sum=20.567839
<[process 0]>[data_proc_B] n=2,avg=0.753088,stddev=0.305515,min=0.537056,max=0.969120,sum=1.506176
NNZ C: 929023247
<Timer>[spgemm] n=31,avg=1334.762817,stddev=180.480331,min=1231.517212,max=2296.719482,sum=41377.648438
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=65.856239,stddev=41.963562,min=36.183521,max=95.528961,sum=131.712479
<[process 0]>[comp_time] n=2,avg=295.085144,stddev=159.921127,min=182.003845,max=408.166473,sum=590.170288
<[process 0]>[data_proc_A] n=2,avg=28.190432,stddev=28.379126,min=8.123360,max=48.257504,sum=56.380863
<[process 0]>[data_proc_B] n=2,avg=2.805968,stddev=2.248147,min=1.216288,max=4.395648,sum=5.611936
NNZ C: 929023247
<Timer>[spgemm] n=32,avg=1334.269653,stddev=177.567429,min=1231.517212,max=2296.719482,sum=42696.628906
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=43.514881,stddev=36.295509,min=17.850080,max=69.179680,sum=87.029762
<[process 0]>[comp_time] n=2,avg=272.476349,stddev=144.038940,min=170.625443,max=374.327271,sum=544.952698
<[process 0]>[data_proc_A] n=2,avg=3.985168,stddev=3.214655,min=1.712064,max=6.258272,sum=7.970336
<[process 0]>[data_proc_B] n=2,avg=0.792528,stddev=0.501084,min=0.438208,max=1.146848,sum=1.585056
NNZ C: 929023247
<Timer>[spgemm] n=33,avg=1332.673096,stddev=175.011398,min=1231.517212,max=2296.719482,sum=43978.210938
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=39.754433,stddev=30.737082,min=18.020033,max=61.488831,sum=79.508865
<[process 0]>[comp_time] n=2,avg=265.819763,stddev=147.436752,min=161.566238,max=370.073303,sum=531.639526
<[process 0]>[data_proc_A] n=2,avg=4.381824,stddev=3.148379,min=2.155584,max=6.608064,sum=8.763648
<[process 0]>[data_proc_B] n=2,avg=0.815536,stddev=0.198737,min=0.675008,max=0.956064,sum=1.631072
NNZ C: 929023247
<Timer>[spgemm] n=34,avg=1332.031982,stddev=172.379883,min=1231.517212,max=2296.719482,sum=45289.085938
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=30.081568,stddev=39.931190,min=1.845952,max=58.317184,sum=60.163136
<[process 0]>[comp_time] n=2,avg=286.793213,stddev=120.690811,min=201.451935,max=372.134521,sum=573.586426
<[process 0]>[data_proc_A] n=2,avg=9.012033,stddev=7.289513,min=3.857568,max=14.166496,sum=18.024065
<[process 0]>[data_proc_B] n=2,avg=3.982816,stddev=4.897116,min=0.520032,max=7.445600,sum=7.965632
NNZ C: 929023247
<Timer>[spgemm] n=35,avg=1331.305786,stddev=169.880264,min=1231.517212,max=2296.719482,sum=46595.703125
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=36.267426,stddev=22.583881,min=20.298208,max=52.236641,sum=72.534851
<[process 0]>[comp_time] n=2,avg=292.978058,stddev=177.007370,min=167.814941,max=418.141174,sum=585.956116
<[process 0]>[data_proc_A] n=2,avg=9.415520,stddev=3.714109,min=6.789248,max=12.041792,sum=18.831039
<[process 0]>[data_proc_B] n=2,avg=4.253264,stddev=4.618867,min=0.987232,max=7.519296,sum=8.506528
NNZ C: 929023247
<Timer>[spgemm] n=36,avg=1331.891602,stddev=167.472717,min=1231.517212,max=2296.719482,sum=47948.097656
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 1]>[data_proc_A] n=2,avg=9.809248,stddev=9.521346,min=3.076640,max=16.541857,sum=19.618496
<[process 1]>[data_proc_B] n=2,avg=1.332176,stddev=0.422703,min=1.033280,max=1.631072,sum=2.664352
<[process 1]>[comm_wait] n=2,avg=166.013000,stddev=2.303826,min=164.383942,max=167.642044,sum=332.026001
<[process 1]>[comp_time] n=2,avg=66.207397,stddev=51.144665,min=30.042656,max=102.372131,sum=132.414795
<[process 1]>[data_proc_A] n=2,avg=4.513824,stddev=2.307634,min=2.882080,max=6.145568,sum=9.027648
<[process 1]>[data_proc_B] n=2,avg=1.066864,stddev=0.205118,min=0.921824,max=1.211904,sum=2.133728
<[process 1]>[comm_wait] n=2,avg=124.347153,stddev=105.046127,min=50.068321,max=198.625977,sum=248.694305
<[process 1]>[comp_time] n=2,avg=73.566742,stddev=43.238663,min=42.992386,max=104.141090,sum=147.133484
<[process 1]>[data_proc_A] n=2,avg=7.167504,stddev=2.636931,min=5.302912,max=9.032096,sum=14.335009
<[process 1]>[data_proc_B] n=2,avg=2.587184,stddev=2.751969,min=0.641248,max=4.533120,sum=5.174368
<[process 1]>[comm_wait] n=2,avg=175.379562,stddev=9.858473,min=168.408569,max=182.350555,sum=350.759125
<[process 1]>[comp_time] n=2,avg=86.761124,stddev=81.135719,min=29.389503,max=144.132736,sum=173.522247
<[process 1]>[data_proc_A] n=2,avg=3.828272,stddev=2.274485,min=2.219968,max=5.436576,sum=7.656544
<[process 1]>[data_proc_B] n=2,avg=1.017296,stddev=0.433790,min=0.710560,max=1.324032,sum=2.034592
<[process 1]>[comm_wait] n=2,avg=111.656815,stddev=97.021400,min=43.052319,max=180.261307,sum=223.313629
<[process 1]>[comp_time] n=2,avg=65.314636,stddev=50.822376,min=29.377792,max=101.251488,sum=130.629272
<[process 1]>[data_proc_A] n=2,avg=4.358032,stddev=1.476779,min=3.313792,max=5.402272,sum=8.716064
<[process 1]>[data_proc_B] n=2,avg=0.902208,stddev=0.501695,min=0.547456,max=1.256960,sum=1.804416
<[process 1]>[comm_wait] n=2,avg=166.688110,stddev=16.409374,min=155.084930,max=178.291290,sum=333.376221
<[process 1]>[comp_time] n=2,avg=66.478531,stddev=52.714321,min=29.203873,max=103.753181,sum=132.957062
<[process 1]>[data_proc_A] n=2,avg=3.587280,stddev=2.246653,min=1.998656,max=5.175904,sum=7.174560
<[process 1]>[data_proc_B] n=2,avg=0.910960,stddev=0.479588,min=0.571840,max=1.250080,sum=1.821920
<[process 1]>[comm_wait] n=2,avg=171.273132,stddev=18.831692,min=157.957123,max=184.589157,sum=342.546265
<[process 1]>[comp_time] n=2,avg=71.571907,stddev=59.691128,min=29.363905,max=113.779907,sum=143.143814
<[process 1]>[data_proc_A] n=2,avg=4.684048,stddev=0.736047,min=4.163584,max=5.204512,sum=9.368096
<[process 1]>[data_proc_B] n=2,avg=0.874720,stddev=0.461961,min=0.548064,max=1.201376,sum=1.749440
<[process 1]>[comm_wait] n=2,avg=172.023926,stddev=7.351737,min=166.825470,max=177.222397,sum=344.047852
<[process 1]>[comp_time] n=2,avg=75.106018,stddev=51.326809,min=38.812481,max=111.399551,sum=150.212036
<[process 1]>[data_proc_A] n=2,avg=7.777584,stddev=0.469723,min=7.445440,max=8.109728,sum=15.555168
<[process 1]>[data_proc_B] n=2,avg=3.654064,stddev=4.286606,min=0.622976,max=6.685152,sum=7.308128
<[process 1]>[comm_wait] n=2,avg=193.772583,stddev=6.507776,min=189.170883,max=198.374268,sum=387.545166
<[process 1]>[comp_time] n=2,avg=90.198318,stddev=85.003540,min=30.091743,max=150.304901,sum=180.396637
<[process 1]>[data_proc_A] n=2,avg=4.818880,stddev=3.699492,min=2.202944,max=7.434816,sum=9.637760
<[process 1]>[data_proc_B] n=2,avg=3.322432,stddev=3.028589,min=1.180896,max=5.463968,sum=6.644864
<[process 1]>[comm_wait] n=2,avg=135.489120,stddev=46.957134,min=102.285408,max=168.692825,sum=270.978241
<[process 1]>[comp_time] n=2,avg=65.518608,stddev=51.246006,min=29.282207,max=101.755005,sum=131.037216
<[process 1]>[data_proc_A] n=2,avg=3.864592,stddev=2.231900,min=2.286400,max=5.442784,sum=7.729184
<[process 2]>[data_proc_A] n=2,avg=18.557983,stddev=11.552881,min=10.388864,max=26.727104,sum=37.115967
<[process 2]>[data_proc_B] n=2,avg=3.014960,stddev=2.844561,min=1.003552,max=5.026368,sum=6.029920
<[process 2]>[comm_wait] n=2,avg=147.380798,stddev=73.603088,min=95.335548,max=199.426041,sum=294.761597
<[process 2]>[comp_time] n=2,avg=71.431137,stddev=47.944237,min=37.529442,max=105.332832,sum=142.862274
<[process 2]>[data_proc_A] n=2,avg=7.877776,stddev=0.190229,min=7.743264,max=8.012288,sum=15.755552
<[process 2]>[data_proc_B] n=2,avg=13.651056,stddev=14.463604,min=3.423744,max=23.878368,sum=27.302113
<[process 2]>[comm_wait] n=2,avg=93.856354,stddev=2.779415,min=91.891006,max=95.821693,sum=187.712708
<[process 2]>[comp_time] n=2,avg=69.766861,stddev=52.196766,min=32.858177,max=106.675552,sum=139.533722
<[process 2]>[data_proc_A] n=2,avg=4.206320,stddev=4.301992,min=1.164352,max=7.248288,sum=8.412641
<[process 2]>[data_proc_B] n=2,avg=3.222080,stddev=2.834672,min=1.217664,max=5.226496,sum=6.444160
<[process 2]>[comm_wait] n=2,avg=150.516296,stddev=80.078796,min=93.892029,max=207.140549,sum=301.032593
<[process 2]>[comp_time] n=2,avg=100.063698,stddev=63.404129,min=55.230209,max=144.897186,sum=200.127396
<[process 2]>[data_proc_A] n=2,avg=21.887472,stddev=8.203638,min=16.086624,max=27.688320,sum=43.774944
<[process 2]>[data_proc_B] n=2,avg=4.073392,stddev=4.327109,min=1.013664,max=7.133120,sum=8.146784
<[process 2]>[comm_wait] n=2,avg=69.633636,stddev=38.002388,min=42.761887,max=96.505379,sum=139.267273
<[process 2]>[comp_time] n=2,avg=69.889458,stddev=52.736702,min=32.598976,max=107.179939,sum=139.778915
<[process 2]>[data_proc_A] n=2,avg=5.441888,stddev=3.505123,min=2.963392,max=7.920384,sum=10.883776
<[process 2]>[data_proc_B] n=2,avg=2.970384,stddev=2.541625,min=1.173184,max=4.767584,sum=5.940768
<[process 2]>[comm_wait] n=2,avg=166.579727,stddev=96.305481,min=98.481468,max=234.677979,sum=333.159454
<[process 2]>[comp_time] n=2,avg=80.167679,stddev=33.966331,min=56.149857,max=104.185501,sum=160.335358
<[process 2]>[data_proc_A] n=2,avg=12.229392,stddev=7.296685,min=7.069856,max=17.388927,sum=24.458784
<[process 2]>[data_proc_B] n=2,avg=3.034480,stddev=2.091022,min=1.555904,max=4.513056,sum=6.068960
<[process 2]>[comm_wait] n=2,avg=110.816963,stddev=32.707756,min=87.689087,max=133.944839,sum=221.633926
<[process 2]>[comp_time] n=2,avg=67.927597,stddev=52.321033,min=30.931040,max=104.924156,sum=135.855194
<[process 2]>[data_proc_A] n=2,avg=10.346640,stddev=2.035766,min=8.907136,max=11.786144,sum=20.693279
<[process 2]>[data_proc_B] n=2,avg=1.142016,stddev=0.568898,min=0.739744,max=1.544288,sum=2.284032
<[process 2]>[comm_wait] n=2,avg=146.227768,stddev=65.722763,min=99.754753,max=192.700775,sum=292.455536
<[process 2]>[comp_time] n=2,avg=77.276718,stddev=41.991261,min=47.584415,max=106.969025,sum=154.553436
<[process 2]>[data_proc_A] n=2,avg=13.601904,stddev=8.806026,min=7.375104,max=19.828705,sum=27.203808
<[process 2]>[data_proc_B] n=2,avg=3.001472,stddev=2.716783,min=1.080416,max=4.922528,sum=6.002944
<[process 2]>[comm_wait] n=2,avg=138.453064,stddev=73.374123,min=86.569725,max=190.336411,sum=276.906128
<[process 2]>[comp_time] n=2,avg=88.293518,stddev=80.251785,min=31.546944,max=145.040100,sum=176.587036
<[process 2]>[data_proc_A] n=2,avg=16.549953,stddev=13.583466,min=6.944992,max=26.154913,sum=33.099907
<[process 2]>[data_proc_B] n=2,avg=2.133504,stddev=2.065566,min=0.672928,max=3.594080,sum=4.267008
<[process 2]>[comm_wait] n=2,avg=197.415405,stddev=132.383224,min=103.806335,max=291.024475,sum=394.830811
<[process 2]>[comp_time] n=2,avg=70.227440,stddev=47.938469,min=36.329823,max=104.125053,sum=140.454880
<[process 3]>[data_proc_A] n=2,avg=9.077440,stddev=8.926471,min=2.765472,max=15.389408,sum=18.154881
<[process 3]>[data_proc_B] n=2,avg=3.251056,stddev=3.553342,min=0.738464,max=5.763648,sum=6.502112
<[process 3]>[comm_wait] n=2,avg=70.652573,stddev=37.263462,min=44.303329,max=97.001823,sum=141.305145
<[process 3]>[comp_time] n=2,avg=425.070709,stddev=586.246704,min=10.531680,max=839.609741,sum=850.141418
<[process 3]>[data_proc_A] n=2,avg=5.940256,stddev=4.796469,min=2.548640,max=9.331872,sum=11.880512
<[process 3]>[data_proc_B] n=2,avg=0.835888,stddev=0.384236,min=0.564192,max=1.107584,sum=1.671776
<[process 3]>[comm_wait] n=2,avg=69.217987,stddev=51.178692,min=33.029182,max=105.406784,sum=138.435974
<[process 3]>[comp_time] n=2,avg=427.009155,stddev=562.133850,min=29.520512,max=824.497803,sum=854.018311
<[process 3]>[data_proc_A] n=2,avg=8.951777,stddev=0.164321,min=8.835584,max=9.067968,sum=17.903553
<[process 3]>[data_proc_B] n=2,avg=2.643424,stddev=2.165670,min=1.112064,max=4.174784,sum=5.286848
<[process 3]>[comm_wait] n=2,avg=83.764977,stddev=33.286945,min=60.227551,max=107.302399,sum=167.529953
<[process 3]>[comp_time] n=2,avg=426.087494,stddev=587.230713,min=10.852672,max=841.322327,sum=852.174988
<[process 3]>[data_proc_A] n=2,avg=5.823040,stddev=5.671925,min=1.812384,max=9.833696,sum=11.646080
<[process 3]>[data_proc_B] n=2,avg=1.087648,stddev=0.439651,min=0.776768,max=1.398528,sum=2.175296
<[process 3]>[comm_wait] n=2,avg=73.766495,stddev=57.024712,min=33.443935,max=114.089058,sum=147.532990
<[process 3]>[comp_time] n=2,avg=428.464233,stddev=588.874084,min=12.067392,max=844.861084,sum=856.928467
<[process 3]>[data_proc_A] n=2,avg=9.494064,stddev=8.842999,min=3.241120,max=15.747008,sum=18.988129
<[process 3]>[data_proc_B] n=2,avg=2.884976,stddev=2.934437,min=0.810016,max=4.959936,sum=5.769952
<[process 3]>[comm_wait] n=2,avg=63.564400,stddev=27.395443,min=44.192898,max=82.935905,sum=127.128799
<[process 3]>[comp_time] n=2,avg=416.817810,stddev=573.190308,min=11.511072,max=822.124573,sum=833.635620
<[process 3]>[data_proc_A] n=2,avg=5.569088,stddev=5.498598,min=1.680992,max=9.457184,sum=11.138176
<[process 3]>[data_proc_B] n=2,avg=0.978368,stddev=0.432003,min=0.672896,max=1.283840,sum=1.956736
<[process 3]>[comm_wait] n=2,avg=109.687355,stddev=108.140129,min=33.220737,max=186.153976,sum=219.374710
<[process 3]>[comp_time] n=2,avg=423.000580,stddev=583.079590,min=10.701056,max=835.300110,sum=846.001160
<[process 3]>[data_proc_A] n=2,avg=6.651360,stddev=4.055874,min=3.783424,max=9.519296,sum=13.302719
<[process 3]>[data_proc_B] n=2,avg=0.905024,stddev=0.330949,min=0.671008,max=1.139040,sum=1.810048
<[process 3]>[comm_wait] n=2,avg=83.460304,stddev=26.981859,min=64.381248,max=102.539360,sum=166.920609
<[process 3]>[comp_time] n=2,avg=425.587036,stddev=571.200317,min=21.687424,max=829.486633,sum=851.174072
<[process 3]>[data_proc_A] n=2,avg=7.903328,stddev=0.642166,min=7.449248,max=8.357408,sum=15.806656
<[process 3]>[data_proc_B] n=2,avg=3.778320,stddev=3.531393,min=1.281248,max=6.275392,sum=7.556640
<[process 3]>[comm_wait] n=2,avg=56.625633,stddev=7.756633,min=51.140865,max=62.110401,sum=113.251266
<[process 3]>[comp_time] n=2,avg=430.648621,stddev=593.948120,min=10.663872,max=850.633362,sum=861.297241
<[process 3]>[data_proc_A] n=2,avg=10.421456,stddev=12.079761,min=1.879776,max=18.963137,sum=20.842913
<[process 3]>[data_proc_B] n=2,avg=2.645696,stddev=2.867437,min=0.618112,max=4.673280,sum=5.291392
<[process 3]>[comm_wait] n=2,avg=69.578255,stddev=32.444660,min=46.636417,max=92.520096,sum=139.156509
<[process 3]>[comp_time] n=2,avg=417.061340,stddev=574.623962,min=10.740864,max=823.381836,sum=834.122681
<[process 0]>[comm_wait] n=2,avg=35.761230,stddev=27.636110,min=16.219551,max=55.302914,sum=71.522461
<[process 0]>[comp_time] n=2,avg=277.995361,stddev=129.167511,min=186.660156,max=369.330597,sum=555.990723
<[process 0]>[data_proc_A] n=2,avg=4.468560,stddev=3.874877,min=1.728608,max=7.208512,sum=8.937119
<[process 0]>[data_proc_B] n=2,avg=0.700896,stddev=0.338732,min=0.461376,max=0.940416,sum=1.401792
NNZ C: 929023247
<[process 3]>[data_proc_A] n=2,avg=5.732288,stddev=5.304862,min=1.981184,max=9.483392,sum=11.464576
<Timer>[spgemm] n=37,avg=1330.158447,stddev=165.466492,min=1231.517212,max=2296.719482,sum=49215.863281
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=40.547134,stddev=31.626114,min=18.184095,max=62.910175,sum=81.094269
<[process 0]>[comp_time] n=2,avg=265.516174,stddev=152.384552,min=157.764008,max=373.268311,sum=531.032349
<[process 0]>[data_proc_A] n=2,avg=4.032400,stddev=3.215967,min=1.758368,max=6.306432,sum=8.064800
<[process 0]>[data_proc_B] n=2,avg=0.873536,stddev=0.433903,min=0.566720,max=1.180352,sum=1.747072
NNZ C: 929023247
<Timer>[spgemm] n=38,avg=1328.818115,stddev=163.424149,min=1231.517212,max=2296.719482,sum=50495.089844
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=14.011440,stddev=1.272452,min=13.111680,max=14.911200,sum=28.022881
<[process 0]>[comp_time] n=2,avg=305.868530,stddev=88.803955,min=243.074661,max=368.662415,sum=611.737061
<[process 0]>[data_proc_A] n=2,avg=7.676816,stddev=8.336099,min=1.782304,max=13.571328,sum=15.353632
<[process 0]>[data_proc_B] n=2,avg=3.237104,stddev=3.774774,min=0.567936,max=5.906272,sum=6.474208
NNZ C: 929023247
<Timer>[spgemm] n=39,avg=1329.107422,stddev=161.269608,min=1231.517212,max=2296.719482,sum=51835.187500
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=40.681263,stddev=20.814348,min=25.963297,max=55.399231,sum=81.362526
<[process 0]>[comp_time] n=2,avg=288.996185,stddev=113.331718,min=208.858566,max=369.133820,sum=577.992371
<[process 0]>[data_proc_A] n=2,avg=14.230160,stddev=17.279043,min=2.012032,max=26.448288,sum=28.460320
<[process 0]>[data_proc_B] n=2,avg=0.847360,stddev=0.465129,min=0.518464,max=1.176256,sum=1.694720
NNZ C: 929023247
<Timer>[spgemm] n=40,avg=1328.932007,stddev=159.192474,min=1231.517212,max=2296.719482,sum=53157.281250
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=38.254784,stddev=31.083193,min=16.275648,max=60.233921,sum=76.509567
<[process 0]>[comp_time] n=2,avg=275.157898,stddev=132.156952,min=181.708801,max=368.606964,sum=550.315796
<[process 0]>[data_proc_A] n=2,avg=3.744784,stddev=2.897780,min=1.695744,max=5.793824,sum=7.489568
<[process 0]>[data_proc_B] n=2,avg=0.833744,stddev=0.551226,min=0.443968,max=1.223520,sum=1.667488
NNZ C: 929023247
<Timer>[spgemm] n=41,avg=1327.720825,stddev=157.381165,min=1231.517212,max=2296.719482,sum=54436.554688
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=77.650093,stddev=86.366432,min=16.579807,max=138.720383,sum=155.300186
<[process 0]>[comp_time] n=2,avg=283.896454,stddev=121.987045,min=197.638596,max=370.154327,sum=567.792908
<[process 0]>[data_proc_A] n=2,avg=4.770032,stddev=3.890852,min=2.018784,max=7.521280,sum=9.540064
<[process 0]>[data_proc_B] n=2,avg=0.754192,stddev=0.251323,min=0.576480,max=0.931904,sum=1.508384
NNZ C: 929023247
<Timer>[spgemm] n=42,avg=1327.189209,stddev=155.488220,min=1231.517212,max=2296.719482,sum=55741.945312
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=78.653297,stddev=87.060593,min=17.092159,max=140.214432,sum=157.306595
<[process 0]>[comp_time] n=2,avg=284.100098,stddev=119.961960,min=199.274170,max=368.925995,sum=568.200195
<[process 0]>[data_proc_A] n=2,avg=4.248912,stddev=3.715128,min=1.621920,max=6.875904,sum=8.497824
<[process 0]>[data_proc_B] n=2,avg=3.140336,stddev=2.780027,min=1.174560,max=5.106112,sum=6.280672
NNZ C: 929023247
<Timer>[spgemm] n=43,avg=1326.833740,stddev=153.643707,min=1231.517212,max=2296.719482,sum=57053.851562
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=81.726143,stddev=91.246368,min=17.205215,max=146.247070,sum=163.452286
<[process 0]>[comp_time] n=2,avg=307.970215,stddev=181.433609,min=179.677277,max=436.263153,sum=615.940430
<[process 0]>[data_proc_A] n=2,avg=7.626160,stddev=0.559916,min=7.230240,max=8.022080,sum=15.252320
<[process 0]>[data_proc_B] n=2,avg=5.654896,stddev=1.795327,min=4.385408,max=6.924384,sum=11.309792
NNZ C: 929023247
<Timer>[spgemm] n=44,avg=1327.289185,stddev=151.876709,min=1231.517212,max=2296.719482,sum=58400.726562
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=63.920929,stddev=67.104958,min=16.470560,max=111.371300,sum=127.841858
<[process 0]>[comp_time] n=2,avg=293.604553,stddev=115.965965,min=211.604218,max=375.604858,sum=587.209106
<[process 0]>[data_proc_A] n=2,avg=11.262689,stddev=13.578939,min=1.660928,max=20.864449,sum=22.525377
<[process 0]>[data_proc_B] n=2,avg=0.828560,stddev=0.540592,min=0.446304,max=1.210816,sum=1.657120
NNZ C: 929023247
<Timer>[spgemm] n=45,avg=1325.896484,stddev=150.431274,min=1231.517212,max=2296.719482,sum=59665.343750
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 2]>[data_proc_A] n=2,avg=4.328432,stddev=4.309369,min=1.281248,max=7.375616,sum=8.656864
<[process 2]>[data_proc_B] n=2,avg=3.629664,stddev=3.451677,min=1.188960,max=6.070368,sum=7.259328
<[process 2]>[comm_wait] n=2,avg=126.332031,stddev=71.336647,min=75.889404,max=176.774658,sum=252.664062
<[process 2]>[comp_time] n=2,avg=89.186607,stddev=23.373831,min=72.658813,max=105.714401,sum=178.373215
<[process 2]>[data_proc_A] n=2,avg=11.477488,stddev=6.810016,min=6.662080,max=16.292896,sum=22.954975
<[process 2]>[data_proc_B] n=2,avg=1.798912,stddev=1.783900,min=0.537504,max=3.060320,sum=3.597824
<[process 2]>[comm_wait] n=2,avg=155.467056,stddev=77.432274,min=100.714172,max=210.219940,sum=310.934113
<[process 2]>[comp_time] n=2,avg=103.753952,stddev=18.582355,min=90.614243,max=116.893661,sum=207.507904
<[process 2]>[data_proc_A] n=2,avg=18.921631,stddev=24.813587,min=1.375776,max=36.467487,sum=37.843262
<[process 2]>[data_proc_B] n=2,avg=1.127104,stddev=0.121781,min=1.040992,max=1.213216,sum=2.254208
<[process 2]>[comm_wait] n=2,avg=157.344742,stddev=71.419647,min=106.843422,max=207.846054,sum=314.689484
<[process 2]>[comp_time] n=2,avg=106.865181,stddev=75.787323,min=53.275455,max=160.454910,sum=213.730362
<[process 2]>[data_proc_A] n=2,avg=22.503471,stddev=4.767438,min=19.132383,max=25.874559,sum=45.006943
<[process 2]>[data_proc_B] n=2,avg=4.773888,stddev=5.125020,min=1.149952,max=8.397824,sum=9.547776
<[process 2]>[comm_wait] n=2,avg=108.980438,stddev=17.753450,min=96.426849,max=121.534019,sum=217.960876
<[process 2]>[comp_time] n=2,avg=67.910271,stddev=52.383465,min=30.869568,max=104.950974,sum=135.820541
<[process 2]>[data_proc_A] n=2,avg=13.178608,stddev=6.078516,min=8.880448,max=17.476768,sum=26.357216
<[process 2]>[data_proc_B] n=2,avg=0.815488,stddev=0.336289,min=0.577696,max=1.053280,sum=1.630976
<[process 2]>[comm_wait] n=2,avg=139.226974,stddev=52.618431,min=102.020126,max=176.433823,sum=278.453949
<[process 2]>[comp_time] n=2,avg=77.453812,stddev=59.519135,min=35.367424,max=119.540192,sum=154.907623
<[process 2]>[data_proc_A] n=2,avg=4.409952,stddev=4.608797,min=1.151040,max=7.668864,sum=8.819903
<[process 2]>[data_proc_B] n=2,avg=4.095888,stddev=2.211581,min=2.532064,max=5.659712,sum=8.191776
<[process 2]>[comm_wait] n=2,avg=123.648384,stddev=45.013351,min=91.819138,max=155.477631,sum=247.296768
<[process 2]>[comp_time] n=2,avg=73.833389,stddev=60.069481,min=31.357857,max=116.308929,sum=147.666779
<[process 2]>[data_proc_A] n=2,avg=16.108017,stddev=1.509905,min=15.040352,max=17.175680,sum=32.216034
<[process 2]>[data_proc_B] n=2,avg=1.912528,stddev=1.303452,min=0.990848,max=2.834208,sum=3.825056
<[process 2]>[comm_wait] n=2,avg=166.300781,stddev=85.234474,min=106.030914,max=226.570663,sum=332.601562
<[process 2]>[comp_time] n=2,avg=96.140991,stddev=82.112579,min=38.078625,max=154.203354,sum=192.281982
<[process 2]>[data_proc_A] n=2,avg=13.254176,stddev=0.548941,min=12.866016,max=13.642336,sum=26.508352
<[process 2]>[data_proc_B] n=2,avg=5.718992,stddev=3.338472,min=3.358336,max=8.079648,sum=11.437984
<[process 2]>[comm_wait] n=2,avg=158.139465,stddev=75.276863,min=104.910690,max=211.368256,sum=316.278931
<[process 2]>[comp_time] n=2,avg=82.431358,stddev=31.718977,min=60.002655,max=104.860062,sum=164.862717
<[process 2]>[data_proc_A] n=2,avg=12.148096,stddev=4.613369,min=8.885952,max=15.410240,sum=24.296192
<[process 2]>[data_proc_B] n=2,avg=7.670960,stddev=7.891470,min=2.090848,max=13.251072,sum=15.341920
<[process 2]>[comm_wait] n=2,avg=155.850815,stddev=79.993713,min=99.286720,max=212.414917,sum=311.701630
<[process 2]>[comp_time] n=2,avg=71.213165,stddev=53.390408,min=33.460449,max=108.965889,sum=142.426331
<[process 2]>[data_proc_A] n=2,avg=14.481088,stddev=9.918909,min=7.467360,max=21.494816,sum=28.962175
<[process 3]>[data_proc_B] n=2,avg=1.001440,stddev=0.401456,min=0.717568,max=1.285312,sum=2.002880
<[process 3]>[comm_wait] n=2,avg=76.180878,stddev=40.488388,min=47.551266,max=104.810493,sum=152.361755
<[process 3]>[comp_time] n=2,avg=425.258453,stddev=586.042358,min=10.863936,max=839.652954,sum=850.516907
<[process 3]>[data_proc_A] n=2,avg=38.524834,stddev=41.083923,min=9.474112,max=67.575554,sum=77.049667
<[process 3]>[data_proc_B] n=2,avg=3.743600,stddev=3.605883,min=1.193856,max=6.293344,sum=7.487200
<[process 3]>[comm_wait] n=2,avg=80.404274,stddev=22.550594,min=64.458595,max=96.349953,sum=160.808548
<[process 3]>[comp_time] n=2,avg=445.162994,stddev=575.427063,min=38.274624,max=852.051392,sum=890.325989
<[process 3]>[data_proc_A] n=2,avg=8.618720,stddev=1.130873,min=7.819072,max=9.418368,sum=17.237440
<[process 3]>[data_proc_B] n=2,avg=3.239936,stddev=2.839650,min=1.232000,max=5.247872,sum=6.479872
<[process 3]>[comm_wait] n=2,avg=71.832878,stddev=28.064991,min=51.987934,max=91.677826,sum=143.665756
<[process 3]>[comp_time] n=2,avg=430.688141,stddev=590.164124,min=13.379104,max=847.997192,sum=861.376282
<[process 3]>[data_proc_A] n=2,avg=5.713888,stddev=4.916395,min=2.237472,max=9.190304,sum=11.427776
<[process 3]>[data_proc_B] n=2,avg=1.212016,stddev=0.860724,min=0.603392,max=1.820640,sum=2.424032
<[process 3]>[comm_wait] n=2,avg=106.180412,stddev=91.662758,min=41.365055,max=170.995773,sum=212.360825
<[process 3]>[comp_time] n=2,avg=418.499603,stddev=574.364807,min=12.362368,max=824.636841,sum=836.999207
<[process 3]>[data_proc_A] n=2,avg=7.564256,stddev=5.385190,min=3.756352,max=11.372160,sum=15.128511
<[process 3]>[data_proc_B] n=2,avg=1.073760,stddev=0.705840,min=0.574656,max=1.572864,sum=2.147520
<[process 3]>[comm_wait] n=2,avg=61.692146,stddev=35.742741,min=36.418209,max=86.966080,sum=123.384293
<[process 3]>[comp_time] n=2,avg=421.774384,stddev=578.815552,min=12.489984,max=831.058777,sum=843.548767
<[process 3]>[data_proc_A] n=2,avg=6.322320,stddev=5.268455,min=2.596960,max=10.047680,sum=12.644640
<[process 3]>[data_proc_B] n=2,avg=0.902640,stddev=0.274900,min=0.708256,max=1.097024,sum=1.805280
<[process 3]>[comm_wait] n=2,avg=130.324310,stddev=107.741150,min=54.139809,max=206.508804,sum=260.648621
<[process 3]>[comp_time] n=2,avg=419.783936,stddev=578.466370,min=10.746464,max=828.821411,sum=839.567871
<[process 3]>[data_proc_A] n=2,avg=10.072496,stddev=0.811034,min=9.499008,max=10.645984,sum=20.144993
<[process 3]>[data_proc_B] n=2,avg=3.854784,stddev=3.475481,min=1.397248,max=6.312320,sum=7.709568
<[process 3]>[comm_wait] n=2,avg=66.085487,stddev=47.231133,min=32.688030,max=99.482941,sum=132.170975
<[process 3]>[comp_time] n=2,avg=429.634521,stddev=593.231262,min=10.156704,max=849.112366,sum=859.269043
<[process 3]>[data_proc_A] n=2,avg=7.996288,stddev=8.797902,min=1.775232,max=14.217344,sum=15.992577
<[process 3]>[data_proc_B] n=2,avg=3.885024,stddev=4.681794,min=0.574496,max=7.195552,sum=7.770048
<[process 3]>[comm_wait] n=2,avg=51.639534,stddev=5.191162,min=47.968830,max=55.310242,sum=103.279068
<[process 3]>[comp_time] n=2,avg=418.760254,stddev=573.417603,min=13.292800,max=824.227722,sum=837.520508
<[process 3]>[data_proc_A] n=2,avg=5.081328,stddev=5.277461,min=1.349600,max=8.813056,sum=10.162656
<[process 3]>[data_proc_B] n=2,avg=1.141616,stddev=0.703600,min=0.644096,max=1.639136,sum=2.283232
<[process 3]>[comm_wait] n=2,avg=69.717712,stddev=44.851227,min=38.003105,max=101.432320,sum=139.435425
<[process 3]>[comp_time] n=2,avg=424.937988,stddev=581.575867,min=13.701760,max=836.174194,sum=849.875977
<[process 3]>[data_proc_A] n=2,avg=6.137648,stddev=4.971267,min=2.622432,max=9.652864,sum=12.275296
<[process 0]>[comm_wait] n=2,avg=83.389793,stddev=80.208725,min=26.673664,max=140.105927,sum=166.779587
<[process 0]>[comp_time] n=2,avg=299.812805,stddev=102.428864,min=227.384644,max=372.240936,sum=599.625610
<[process 0]>[data_proc_A] n=2,avg=4.984960,stddev=3.407553,min=2.575456,max=7.394464,sum=9.969920
<[process 0]>[data_proc_B] n=2,avg=1.946640,stddev=1.895431,min=0.606368,max=3.286912,sum=3.893280
NNZ C: 929023247
<Timer>[spgemm] n=46,avg=1325.477783,stddev=148.777527,min=1231.517212,max=2296.719482,sum=60971.976562
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 1]>[data_proc_B] n=2,avg=0.898176,stddev=0.499432,min=0.545024,max=1.251328,sum=1.796352
<[process 1]>[comm_wait] n=2,avg=157.997345,stddev=19.349699,min=144.315033,max=171.679642,sum=315.994690
<[process 1]>[comp_time] n=2,avg=67.538239,stddev=54.456135,min=29.031937,max=106.044540,sum=135.076477
<[process 1]>[data_proc_A] n=2,avg=3.896032,stddev=2.876352,min=1.862144,max=5.929920,sum=7.792064
<[process 1]>[data_proc_B] n=2,avg=0.946832,stddev=0.365772,min=0.688192,max=1.205472,sum=1.893664
<[process 1]>[comm_wait] n=2,avg=123.267075,stddev=66.504013,min=76.241631,max=170.292511,sum=246.534149
<[process 1]>[comp_time] n=2,avg=83.564781,stddev=46.596344,min=50.616192,max=116.513374,sum=167.129562
<[process 1]>[data_proc_A] n=2,avg=8.586945,stddev=1.139426,min=7.781248,max=9.392640,sum=17.173889
<[process 1]>[data_proc_B] n=2,avg=3.588176,stddev=2.941632,min=1.508128,max=5.668224,sum=7.176352
<[process 1]>[comm_wait] n=2,avg=180.015152,stddev=15.869031,min=168.794052,max=191.236252,sum=360.030304
<[process 1]>[comp_time] n=2,avg=91.969200,stddev=85.101738,min=31.793184,max=152.145218,sum=183.938400
<[process 1]>[data_proc_A] n=2,avg=3.940960,stddev=1.871830,min=2.617376,max=5.264544,sum=7.881920
<[process 1]>[data_proc_B] n=2,avg=0.971456,stddev=0.312168,min=0.750720,max=1.192192,sum=1.942912
<[process 1]>[comm_wait] n=2,avg=192.382050,stddev=24.382042,min=175.141342,max=209.622757,sum=384.764099
<[process 1]>[comp_time] n=2,avg=65.919472,stddev=50.688648,min=30.077185,max=101.761757,sum=131.838943
<[process 1]>[data_proc_A] n=2,avg=4.692336,stddev=0.787864,min=4.135232,max=5.249440,sum=9.384672
<[process 1]>[data_proc_B] n=2,avg=0.906912,stddev=0.327871,min=0.675072,max=1.138752,sum=1.813824
<[process 1]>[comm_wait] n=2,avg=160.579483,stddev=4.848692,min=157.150940,max=164.008026,sum=321.158966
<[process 1]>[comp_time] n=2,avg=67.667488,stddev=54.138359,min=29.385887,max=105.949089,sum=135.334976
<[process 1]>[data_proc_A] n=2,avg=5.581136,stddev=3.715988,min=2.953536,max=8.208736,sum=11.162272
<[process 1]>[data_proc_B] n=2,avg=2.281200,stddev=1.495016,min=1.224064,max=3.338336,sum=4.562400
<[process 1]>[comm_wait] n=2,avg=195.658722,stddev=13.055202,min=186.427292,max=204.890137,sum=391.317444
<[process 1]>[comp_time] n=2,avg=72.754799,stddev=60.236107,min=30.161440,max=115.348160,sum=145.509598
<[process 1]>[data_proc_A] n=2,avg=8.499104,stddev=3.927622,min=5.721856,max=11.276352,sum=16.998207
<[process 1]>[data_proc_B] n=2,avg=3.581120,stddev=4.263277,min=0.566528,max=6.595712,sum=7.162240
<[process 1]>[comm_wait] n=2,avg=165.916321,stddev=11.201658,min=157.995544,max=173.837082,sum=331.832642
<[process 1]>[comp_time] n=2,avg=91.378448,stddev=83.721458,min=32.178432,max=150.578461,sum=182.756897
<[process 1]>[data_proc_A] n=2,avg=3.708256,stddev=2.256813,min=2.112448,max=5.304064,sum=7.416512
<[process 1]>[data_proc_B] n=2,avg=0.945232,stddev=0.346358,min=0.700320,max=1.190144,sum=1.890464
<[process 1]>[comm_wait] n=2,avg=170.748657,stddev=15.273824,min=159.948441,max=181.548889,sum=341.497314
<[process 1]>[comp_time] n=2,avg=67.097900,stddev=50.608593,min=31.312223,max=102.883583,sum=134.195801
<[process 1]>[data_proc_A] n=2,avg=3.464928,stddev=2.443263,min=1.737280,max=5.192576,sum=6.929856
<[process 1]>[data_proc_B] n=2,avg=0.889776,stddev=0.502578,min=0.534400,max=1.245152,sum=1.779552
<[process 1]>[comm_wait] n=2,avg=170.992966,stddev=6.655539,min=166.286789,max=175.699142,sum=341.985931
<[process 1]>[comp_time] n=2,avg=67.603073,stddev=51.921867,min=30.888767,max=104.317375,sum=135.206146
<[process 1]>[data_proc_A] n=2,avg=4.324272,stddev=1.888642,min=2.988800,max=5.659744,sum=8.648544
<[process 1]>[data_proc_B] n=2,avg=0.999712,stddev=0.248132,min=0.824256,max=1.175168,sum=1.999424
<[process 0]>[comm_wait] n=2,avg=46.002815,stddev=21.110746,min=31.075264,max=60.930367,sum=92.005630
<[process 0]>[comp_time] n=2,avg=279.142334,stddev=129.342773,min=187.683167,max=370.601471,sum=558.284668
<[process 0]>[data_proc_A] n=2,avg=33.338787,stddev=44.968689,min=1.541120,max=65.136452,sum=66.677574
<[process 0]>[data_proc_B] n=2,avg=2.306768,stddev=1.173661,min=1.476864,max=3.136672,sum=4.613536
NNZ C: 929023247
<Timer>[spgemm] n=47,avg=1324.827515,stddev=147.219009,min=1231.517212,max=2296.719482,sum=62266.894531
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=14.807104,stddev=20.740381,min=0.141440,max=29.472769,sum=29.614208
<[process 0]>[comp_time] n=2,avg=272.695618,stddev=136.600861,min=176.104218,max=369.287018,sum=545.391235
<[process 0]>[data_proc_A] n=2,avg=7.161424,stddev=6.611528,min=2.486368,max=11.836480,sum=14.322848
<[process 0]>[data_proc_B] n=2,avg=3.586912,stddev=4.151905,min=0.651072,max=6.522752,sum=7.173824
NNZ C: 929023247
<Timer>[spgemm] n=48,avg=1325.043823,stddev=145.652145,min=1231.517212,max=2296.719482,sum=63602.105469
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=68.955185,stddev=71.112106,min=18.671328,max=119.239037,sum=137.910370
<[process 0]>[comp_time] n=2,avg=306.584045,stddev=140.364182,min=207.331589,max=405.836517,sum=613.168091
<[process 0]>[data_proc_A] n=2,avg=7.302256,stddev=0.725458,min=6.789280,max=7.815232,sum=14.604511
<[process 0]>[data_proc_B] n=2,avg=6.001152,stddev=6.799992,min=1.192832,max=10.809472,sum=12.002304
NNZ C: 929023247
<Timer>[spgemm] n=49,avg=1324.556885,stddev=144.167221,min=1231.517212,max=2296.719482,sum=64903.289062
STARTING spgemm round: 
Iteration 0
Iteration 1
<[process 0]>[comm_wait] n=2,avg=44.595119,stddev=22.748768,min=28.509312,max=60.680927,sum=89.190239
<[process 0]>[comp_time] n=2,avg=271.929840,stddev=137.950058,min=174.384415,max=369.475250,sum=543.859680
<[process 0]>[data_proc_A] n=2,avg=40.230610,stddev=54.731445,min=1.529632,max=78.931587,sum=80.461220
<[process 0]>[data_proc_B] n=2,avg=20.014576,stddev=27.679327,min=0.442336,max=39.586815,sum=40.029152
NNZ C: 929023247
<Timer>[spgemm] n=50,avg=1323.026367,stddev=143.098404,min=1231.517212,max=2296.719482,sum=66151.320312
STARTING spgemm round: 
Done spgemm
<[process 2]>[data_proc_B] n=2,avg=6.403200,stddev=0.979134,min=5.710848,max=7.095552,sum=12.806400
<[process 2]>[comm_wait] n=2,avg=78.421822,stddev=22.303165,min=62.651104,max=94.192543,sum=156.843643
<[process 2]>[comp_time] n=2,avg=69.770493,stddev=51.968159,min=33.023457,max=106.517532,sum=139.540985
<[process 2]>[data_proc_A] n=2,avg=8.216832,stddev=0.821058,min=7.636256,max=8.797408,sum=16.433664
<[process 2]>[data_proc_B] n=2,avg=3.942128,stddev=0.915573,min=3.294720,max=4.589536,sum=7.884256
<[process 2]>[comm_wait] n=2,avg=135.723267,stddev=61.388584,min=92.314980,max=179.131546,sum=271.446533
<[process 2]>[comp_time] n=2,avg=93.576576,stddev=88.592041,min=30.932545,max=156.220612,sum=187.153152
<[process 2]>[data_proc_A] n=2,avg=5.772784,stddev=5.073089,min=2.185568,max=9.360000,sum=11.545568
<[process 2]>[data_proc_B] n=2,avg=0.961696,stddev=0.439967,min=0.650592,max=1.272800,sum=1.923392
<[process 2]>[comm_wait] n=2,avg=135.937622,stddev=3.592147,min=133.397598,max=138.477661,sum=271.875244
<[process 2]>[comp_time] n=2,avg=77.007492,stddev=64.712509,min=31.248833,max=122.766144,sum=154.014984
<[process 2]>[data_proc_A] n=2,avg=7.779344,stddev=0.381611,min=7.509504,max=8.049184,sum=15.558687
<[process 2]>[data_proc_B] n=2,avg=3.404560,stddev=3.998604,min=0.577120,max=6.232000,sum=6.809120
<[process 2]>[comm_wait] n=2,avg=109.323807,stddev=19.220543,min=95.732834,max=122.914787,sum=218.647614
<[process 2]>[comp_time] n=2,avg=90.503777,stddev=82.017509,min=32.508640,max=148.498917,sum=181.007553
<[process 2]>[data_proc_A] n=2,avg=7.063616,stddev=1.316735,min=6.132544,max=7.994688,sum=14.127232
<[process 2]>[data_proc_B] n=2,avg=4.497312,stddev=0.234556,min=4.331456,max=4.663168,sum=8.994624
<[process 3]>[data_proc_B] n=2,avg=0.872640,stddev=0.386114,min=0.599616,max=1.145664,sum=1.745280
<[process 3]>[comm_wait] n=2,avg=73.111969,stddev=57.523464,min=32.436737,max=113.787201,sum=146.223938
<[process 3]>[comp_time] n=2,avg=430.130859,stddev=578.352905,min=21.173569,max=839.088135,sum=860.261719
<[process 3]>[data_proc_A] n=2,avg=11.656640,stddev=0.897675,min=11.021888,max=12.291392,sum=23.313280
<[process 3]>[data_proc_B] n=2,avg=3.849664,stddev=1.161511,min=3.028352,max=4.670976,sum=7.699328
<[process 3]>[comm_wait] n=2,avg=75.444717,stddev=17.232338,min=63.259617,max=87.629822,sum=150.889435
<[process 3]>[comp_time] n=2,avg=427.566223,stddev=589.370422,min=10.818432,max=844.314026,sum=855.132446
<[process 3]>[data_proc_A] n=2,avg=11.035440,stddev=2.341870,min=9.379488,max=12.691392,sum=22.070881
<[process 3]>[data_proc_B] n=2,avg=0.881360,stddev=0.319341,min=0.655552,max=1.107168,sum=1.762720
<[process 3]>[comm_wait] n=2,avg=134.799271,stddev=73.990028,min=82.480415,max=187.118118,sum=269.598541
<[process 3]>[comp_time] n=2,avg=426.073395,stddev=587.694763,min=10.510464,max=841.636353,sum=852.146790
<[process 3]>[data_proc_A] n=2,avg=10.463056,stddev=9.293781,min=3.891360,max=17.034752,sum=20.926111
<[process 3]>[data_proc_B] n=2,avg=3.002864,stddev=3.318695,min=0.656192,max=5.349536,sum=6.005728
<[process 3]>[comm_wait] n=2,avg=51.917423,stddev=14.478674,min=41.679455,max=62.155392,sum=103.834846
<[process 3]>[comp_time] n=2,avg=418.858582,stddev=574.663025,min=12.510432,max=825.206726,sum=837.717163
<[process 3]>[data_proc_A] n=2,avg=5.619776,stddev=4.036912,min=2.765248,max=8.474304,sum=11.239552
<[process 3]>[data_proc_B] n=2,avg=0.893904,stddev=0.349164,min=0.647008,max=1.140800,sum=1.787808
<[process 1]>[comm_wait] n=2,avg=113.203873,stddev=96.127998,min=45.231106,max=181.176636,sum=226.407745
<[process 1]>[comp_time] n=2,avg=75.858139,stddev=52.513527,min=38.725471,max=112.990814,sum=151.716278
<[process 1]>[data_proc_A] n=2,avg=9.171568,stddev=5.407116,min=5.348160,max=12.994976,sum=18.343136
<[process 1]>[data_proc_B] n=2,avg=2.825472,stddev=3.224181,min=0.545632,max=5.105312,sum=5.650944
<[process 1]>[comm_wait] n=2,avg=161.858093,stddev=41.389420,min=132.591354,max=191.124832,sum=323.716187
<[process 1]>[comp_time] n=2,avg=89.741920,stddev=85.368225,min=29.377472,max=150.106369,sum=179.483841
<[process 1]>[data_proc_A] n=2,avg=5.252528,stddev=1.277929,min=4.348896,max=6.156160,sum=10.505056
<[process 1]>[data_proc_B] n=2,avg=0.901200,stddev=0.414738,min=0.607936,max=1.194464,sum=1.802400
<[process 1]>[comm_wait] n=2,avg=172.274872,stddev=15.494061,min=161.318909,max=183.230820,sum=344.549744
<[process 1]>[comp_time] n=2,avg=79.464516,stddev=71.117477,min=29.176865,max=129.752167,sum=158.929031
<[process 1]>[data_proc_A] n=2,avg=9.797056,stddev=8.359609,min=3.885920,max=15.708192,sum=19.594112
<[process 1]>[data_proc_B] n=2,avg=3.667360,stddev=3.518518,min=1.179392,max=6.155328,sum=7.334720
<[process 1]>[comm_wait] n=2,avg=108.360161,stddev=90.698067,min=44.226944,max=172.493378,sum=216.720322
<[process 1]>[comp_time] n=2,avg=87.029633,stddev=81.917725,min=29.105057,max=144.954208,sum=174.059265
<[process 1]>[data_proc_A] n=2,avg=4.707936,stddev=2.344653,min=3.050016,max=6.365856,sum=9.415873
<[process 1]>[data_proc_B] n=2,avg=2.843824,stddev=2.379658,min=1.161152,max=4.526496,sum=5.687648
'''

from pathlib import Path
import re
import pandas as pd
from typing import Dict, List
# from statistics import geometric_mean, stdev
# import sbatchman as sbm

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_timer_string(timer_str):
  """Parse a timer string like 'n=2,avg=36.2,...' into a dict."""
  parts = timer_str.split(",")
  out = {}
  for p in parts:
    k, v = p.split("=")
    out[k.strip()] = float(v)
  return out

def runs_to_dataframe(runs):
  """
  Convert list of runs into a Pandas DataFrame.
  Each row = one (run, rank, timer_name).
  """
  records = []
  
  for run_id, run in enumerate(runs):
    for key, val in run.items():
      if key == "global_timer":
        # Global timer is a single entry
        rec = parse_timer_string(val)
        rec.update({
          "run": run_id,
          "rank": "global",
          "timer": "global_timer"
        })
        records.append(rec)
      else:
        # Key is rank (int)
        rank = key
        timers = val
        for timer_name, timer_str in timers.items():
          rec = parse_timer_string(timer_str)
          rec.update({
            "run": run_id,
            "rank": rank,
            "timer": timer_name
          })
          records.append(rec)
  
  return pd.DataFrame(records)

def parse_stdout(stdout: str) -> List[Dict[str, Dict[str, str]]]:
  runs = []
  run_i = -1
  for line in stdout.splitlines():
    if line.startswith('STARTING spgemm round'):
      run_i += 1
      runs.append({})
    elif line.startswith('<['):
      m = re.match(r'<\[process (\d+)\]>\[(\w+)\] (.+)', line)
      rank = int(m.group(1))
      timer_name = m.group(2)
      timer_data = m.group(3)
      if not runs[run_i].get(rank): runs[run_i][rank] = {}
      runs[run_i][rank][timer_name] = timer_data
    elif line.startswith('<Timer>[spgemm]'):
      runs[run_i]['global_timer'] = line.split(' ')[-1]

  return runs

# def parse_stdout(job: sbm.Job):
#   pass

def main():
  jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
  data = []

  for job in jobs:
    for res in parse_stdout(job):
      m = re.match(r'(\d+)_nodes__(\d+)_cpus-per-task(\w*)', job.config_name)
      # print(model)
      # print(times)
      data.append({
        'cluster': job.cluster_name,
        'nodes': int(m.group(1)),
        'cpus_per_task': int(m.group(2)),
        'mpi_async': job.config_name.endswith('__mpi-async'),
        # TODO
      })

  df = pd.DataFrame(data)
  path = OUT_DIR / f'hns_spgemm_{sbm.get_cluster_name()}_data.csv'
  df.to_csv(path, index=False)
  print(f'Data saved to {path.resolve().absolute()}')

if __name__ == "__main__":
  import pprint
  runs = parse_stdout(stdout)[36]
  runs = [runs, runs]
  pprint.pprint(runs)
  df = runs_to_dataframe(runs)
  print(df)
  data = []
  for grid in ['2x2x1', '4x4x2']:
    for p in ['hns_main', 'hns_get', 'trilinos']:
      for n in [1,4,16]:
        df1 = df.copy()
        df1['cluster'] = 'test'
        df1['program'] = p
        df1['nodes'] = n
        df1['cpus_per_task'] = 3
        df1['mpi_async'] = True
        df1['grid'] = grid if 'hns' in p else '-'
        # FIXME delete
        if p == 'hns_get':
          df1['avg'] = df1['avg']*.8
        if p == 'trilinos':
          df1['avg'] = df1['avg']*.5
        if grid == '2x2x1':
          df1['avg'] = df1['avg']*.9
        data.append(df1)

  df = pd.concat(data, ignore_index=True)
  path = OUT_DIR / f'hns_spgemm_test_data.csv'
  df.to_csv(path, index=False)
  print(f'Data saved to {path.resolve().absolute()}')
  # main()