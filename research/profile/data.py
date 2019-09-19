import numpy as np

SPECTRE_SIZE = 65

STATUS_A = np.array([
    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, 0.997467, 2.568000, 3.812504, 4.638000,
        4.896029, 5.000000, 4.957244, 4.871000, 4.752040, 4.604000, 4.452305,
        4.286000, 4.095232, 3.900000, 3.725500, 3.551000, 3.360616, 3.165000,
        2.970937, 2.776000, 2.580001, 2.383000],
    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, 0.590581, 1.650000, 2.803982, 3.822000,
        4.424023, 4.782000, 4.935414, 5.000000, 4.970295, 4.906000, 4.798153,
        4.644000, 4.454583, 4.221000, 3.941115, 3.609000, 3.222086, 2.766000,
        2.218671, 1.579000, 0.620097, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0],
    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 1.573586,
        3.602000, 4.398523, 4.819000, 4.948891, 5.000000, 4.972905, 4.912000,
        4.797650, 4.620000, 4.374881, 4.040000, 3.572193, 2.989000, 2.302860,
        1.566000, 0.725914, 0.165000, 0.045596, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0]
])

STATUS_M = np.array([
    [-60.291000, -58.991000, -57.691000, -56.391000, -55.091000, -53.791000,
     -52.491000, -51.191000, -49.891000, -48.591000, -47.291000, -45.991000,
     -44.691000, -43.391000, -42.091000, -40.791000, -39.491000, -38.191000,
     -36.891000, -35.591000, -34.291000, -32.991000, -31.691000, -30.391000,
     -29.091000, -27.791000, -26.491000, -25.191000, -23.891000, -22.591000,
     -21.291000, -19.991000, -18.691000, -17.391000, -16.091000, -14.791000,
     -13.491000, -12.191000, -10.891000,  -9.591000,  -8.291000,  -6.991000,
      -5.691000,  -4.391000,  -3.091000,  -1.791000,  -0.491000,   0.824040,
       2.109000,   3.497183,   4.479000,   4.846277,   5.000000,   4.968707,
       4.899000,   4.759728,   4.578000,   4.418271,   4.252000,   4.067352,
       3.875000,   3.683936,   3.491000,   3.296005,   3.099000 ],
    [ -8.388000,  -7.858000,  -7.328000,  -6.798000,  -6.268000,  -5.738000,
      -5.208000,  -4.678000,  -4.148000,  -3.618000,  -3.088000,  -2.558000,
      -2.028000,  -1.498000,  -0.968000,  -0.438000,   0.092000,   0.622313,
       1.152000,   1.686787,   2.207000,   2.710133,   3.156000,   3.508331,
       3.804000,   4.055549,   4.272000,   4.463102,   4.626000,   4.764237,
       4.872000,   4.957048,   5.000000,   4.998716,   4.995000,   4.934949,
       4.818000,   4.662455,   4.458000,   4.210811,   3.915000,   3.568473,
       3.172000,   2.731815,   2.239000,   1.672818,   1.070000,   0.471963,
      -0.130000,  -0.730000,  -1.330000,  -1.930000,  -2.530000,  -3.130000,
      -3.730000,  -4.330000,  -4.930000,  -5.530000,  -6.130000,  -6.730000,
      -7.330000,  -7.930000,  -8.530000,  -9.130000,  -9.730000 ],
    [ -5.397000,  -4.147000,  -2.897000,  -1.647000,  -0.397000,   0.887106,
       2.103000,   3.281977,   4.111000,   4.433957,   4.632000,   4.771515,
       4.871000,   4.956445,   5.000000,   4.986780,   4.955000,   4.874360,
       4.743000,   4.568359,   4.343000,   4.066481,   3.743000,   3.396307,
       2.990000,   2.495219,   1.852000,   0.839493,  -0.348000,  -1.448000,
      -2.548000,  -3.648000,  -4.748000,  -5.848000,  -6.948000,  -8.048000,
      -9.148000, -10.248000, -11.348000, -12.448000, -13.548000, -14.648000,
     -15.748000, -16.848000, -17.948000, -19.048000, -20.148000, -21.248000,
     -22.348000, -23.465978, -24.548000, -25.499164, -26.478000, -27.692959,
     -28.948000, -30.063899, -31.148000, -32.248000, -33.348000, -34.448000,
     -35.548000, -36.648000, -37.748000, -38.848000, -39.948000 ]
])

ILLUM_A = np.array([9.7951,10.8996,12.0853,13.3543,14.708,16.148,17.6753,19.2907,
    20.995,22.7883,24.6709,26.6425,28.7027,30.8508,33.0859,35.4068,37.8121,
    40.3002,42.8693,45.5174,48.2423,51.0418,53.9132,56.8539,59.8611,62.932,
    66.0635,69.2525,72.4959,75.7903,79.1326,82.5193,85.947,89.4124,92.912,
    96.4423,100,103.582,107.184,110.803,114.436,118.08,121.731,125.386,
    129.043,132.697,136.346,139.988,143.618,147.235,150.836,154.418,157.979,
    161.516,165.028,168.51,171.963,175.383,178.769,182.118,185.429,188.701,
    191.931,195.118,198.261])

D_S0 = np.array([63.40, 64.60, 65.80, 80.30, 94.80, 99.80, 104.80, 105.35, 105.90,
    101.35, 96.80, 105.35, 113.90, 119.75, 125.60, 125.55, 125.50,
    123.40, 121.30, 121.30, 121.30, 117.40, 113.50, 113.30, 113.10,
    111.95, 110.80, 108.65, 106.50, 107.65, 108.80, 107.05, 105.30,
    104.85, 104.40, 102.20, 100.00, 98.00, 96.00, 95.55, 95.10, 92.10,
    89.10, 89.80, 90.50, 90.40, 90.30, 89.35, 88.40, 86.20, 84.00,
    84.55, 85.10, 83.50, 81.90, 82.25, 82.60, 83.75, 84.90, 83.10,
    81.30, 76.60, 71.90, 73.10, 74.30])

D_S1 = np.array([38.50, 36.75, 35.00, 39.20, 43.40, 44.85, 46.30, 45.10, 43.90,
    40.50, 37.10, 36.90, 36.70, 36.30, 35.90, 34.25, 32.60, 30.25,
    27.90, 26.10, 24.30, 22.20, 20.10, 18.15, 16.20, 14.70, 13.20,
    10.90, 8.60, 7.35, 6.10, 5.15, 4.20, 3.05, 1.90, 0.95, 0.00,
    -0.80, -1.60, -2.55, -3.50, -3.50, -3.50, -4.65, -5.80, -6.50,
    -7.20, -7.90, -8.60, -9.05, -9.50, -10.20, -10.90, -10.80, -10.70,
    -11.35, -12.00, -13.00, -14.00, -13.80, -13.60, -12.80, -12.00,
    -12.65, -13.30])

D_S2 = np.array([3.00, 2.10, 1.20, 0.05, -1.10, -0.80, -0.50, -0.60, -0.70, -0.95,
    -1.20, -1.90, -2.60, -2.75, -2.90, -2.85, -2.80, -2.70, -2.60,
    -2.60, -2.60, -2.20, -1.80, -1.65, -1.50, -1.40, -1.30, -1.25,
    -1.20, -1.10, -1.00, -0.75, -0.50, -0.40, -0.30, -0.15, 0.00,
    0.10, 0.20, 0.35, 0.50, 1.30, 2.10, 2.65, 3.20, 3.65, 4.10, 4.40,
    4.70, 4.90, 5.10, 5.90, 6.70, 7.00, 7.30, 7.95, 8.60, 9.20, 9.80,
    10.00, 10.20, 9.25, 8.30, 8.95, 9.60])

V = np.array([0, 0, 4.1276E-04, 1.0561E-03, 2.4477E-03, 4.9696E-03, 9.0909E-03,
    1.4360E-02, 2.0443E-02, 2.6457E-02, 3.3752E-02, 4.2453E-02, 5.1540E-02,
    5.8947E-02, 6.6538E-02, 7.4472E-02, 8.7660E-02, 1.0913E-01, 1.3362E-01,
    1.5773E-01, 1.8348E-01, 2.1158E-01, 2.4342E-01, 2.9118E-01, 3.5522E-01,
    4.3550E-01, 5.2907E-01, 6.2979E-01, 7.2759E-01, 8.0387E-01, 8.6611E-01,
    9.1467E-01, 9.6067E-01, 9.8593E-01, 9.9160E-01, 9.9987E-01, 9.9460E-01,
    9.8535E-01, 9.6525E-01, 9.3132E-01, 8.8215E-01, 8.4131E-01, 7.9168E-01,
    7.3264E-01, 6.6879E-01, 6.0338E-01, 5.3516E-01, 4.6755E-01, 4.0263E-01,
    3.4256E-01, 2.8231E-01, 2.2848E-01, 1.8338E-01, 1.4581E-01, 1.1233E-01,
    8.4474E-02, 6.2710E-02, 4.6031E-02, 3.3432E-02, 2.3981E-02, 1.6971E-02,
    1.1841E-02, 8.1286E-03, 5.6566E-03, 3.9376E-03])