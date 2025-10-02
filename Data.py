# Canonical names used across VotingResult and data mappings
# Always import and use these instead of string literals.
VOTED = "Voted"
NOT_VOTED = "Not Voted"
ANO = "ANO 2011"
SPOLU = "Koalice Spolu (ODS, TOP 09, KDU-ČSL)"
PIRSTAN = "Koalice PIRÁTI a STAROSTOVÉ"
KSCM = "Komunistická strana Čech a Moravy (KSČM)"
SPD = "Svoboda a přímá demokracie – Tomio Okamura (SPD)"
CSSD = "Česká strana sociálně demokratická (ČSSD)"
TRIKOLORA_SVOBODNI_SOUKROMNICI = "Trikolóra, Svobodní a Soukromníci"
PRISAH = "Přísaha Roberta Šlachty"
JINA_STRANA = "Jiná strana"

ELECTION_DATA = (
    {
        VOTED: 0.6543,
        NOT_VOTED: 0.3457
    },
    {
        ANO: 0.2712,
        SPOLU: 0.2779,
        PIRSTAN: 0.1562,
        KSCM: 0.0360,
        SPD: 0.0956,
        CSSD: 0.0465,
        TRIKOLORA_SVOBODNI_SOUKROMNICI: 0.0276,
        PRISAH: 0.0468,
        JINA_STRANA: 0.0422
    }
)

ELECTION_REGION_DATA = {
    'Hlavní město Praha': (
        {
            VOTED: 0.7015,
            NOT_VOTED: 0.2985
        },
        {
            ANO: 0.1746,
            SPOLU: 0.4002,
            PIRSTAN: 0.2264,
            SPD: 0.0459,
            PRISAH: 0.0341,
            CSSD: 0.0400,
            KSCM: 0.0214,
            JINA_STRANA: 0.0574
        }
    ),
    'Středočeský kraj': (
        {
            VOTED: 0.6795,
            NOT_VOTED: 0.3205
        },
        {
            ANO: 0.2492,
            SPOLU: 0.2874,
            PIRSTAN: 0.1946,
            SPD: 0.0779,
            PRISAH: 0.0458,
            CSSD: 0.0456,
            KSCM: 0.0348,
            JINA_STRANA: 0.0647
        }
    ),
    'Jihočeský kraj': (
        {
            VOTED: 0.6634,
            NOT_VOTED: 0.3366
        },
        {
            ANO: 0.2665,
            SPOLU: 0.2909,
            PIRSTAN: 0.1350,
            SPD: 0.0898,
            PRISAH: 0.0451,
            CSSD: 0.0545,
            KSCM: 0.0445,
            JINA_STRANA: 0.0737
        }
    ),
    'Plzeňský kraj': (
        {
            VOTED: 0.6472,
            NOT_VOTED: 0.3528
        },
        {
            ANO: 0.2903,
            SPOLU: 0.2657,
            PIRSTAN: 0.1388,
            SPD: 0.0996,
            PRISAH: 0.0472,
            CSSD: 0.0499,
            KSCM: 0.0414,
            JINA_STRANA: 0.0671
        }
    ),
    'Karlovarský kraj': (
        {
            VOTED: 0.5710,
            NOT_VOTED: 0.4290
        },
        {
            ANO: 0.3306,
            SPOLU: 0.2022,
            PIRSTAN: 0.1423,
            SPD: 0.1276,
            PRISAH: 0.0494,
            CSSD: 0.0378,
            KSCM: 0.0341,
            JINA_STRANA: 0.076
        }
    ),
    'Ústecký kraj': (
        {
            VOTED: 0.5765,
            NOT_VOTED: 0.4235
        },
        {
            ANO: 0.3561,
            SPOLU: 0.1977,
            PIRSTAN: 0.1399,
            SPD: 0.1187,
            PRISAH: 0.0436,
            CSSD: 0.0319,
            KSCM: 0.0392,
            JINA_STRANA: 0.0729
        }
    ),
    'Liberecký kraj': (
        {
            VOTED: 0.6460,
            NOT_VOTED: 0.3540
        },
        {
            ANO: 0.2686,
            SPOLU: 0.2277,
            PIRSTAN: 0.2137,
            SPD: 0.1100,
            PRISAH: 0.0429,
            CSSD: 0.0348,
            KSCM: 0.0301,
            JINA_STRANA: 0.0722
        }
    ),
    'Královéhradecký kraj': (
        {
            VOTED: 0.6786,
            NOT_VOTED: 0.3214
        },
        {
            ANO: 0.2699,
            SPOLU: 0.2858,
            PIRSTAN: 0.1513,
            SPD: 0.0905,
            PRISAH: 0.0473,
            CSSD: 0.0493,
            KSCM: 0.0344,
            JINA_STRANA: 0.0715
        }
    ),
    'Pardubický kraj': (
        {
            VOTED: 0.6789,
            NOT_VOTED: 0.3211
        },
        {
            ANO: 0.2684,
            SPOLU: 0.2852,
            PIRSTAN: 0.1409,
            SPD: 0.0936,
            PRISAH: 0.0495,
            CSSD: 0.0505,
            KSCM: 0.0381,
            JINA_STRANA: 0.0738
        }
    ),
    'Kraj Vysočina': (
        {
            VOTED: 0.6893,
            NOT_VOTED: 0.3107
        },
        {
            ANO: 0.2673,
            SPOLU: 0.2800,
            PIRSTAN: 0.1351,
            SPD: 0.0892,
            PRISAH: 0.0523,
            CSSD: 0.0655,
            KSCM: 0.0467,
            JINA_STRANA: 0.0639
        }
    ),
    'Jihomoravský kraj': (
        {
            VOTED: 0.6639,
            NOT_VOTED: 0.3361
        },
        {
            ANO: 0.2536,
            SPOLU: 0.3003,
            PIRSTAN: 0.1419,
            SPD: 0.0935,
            PRISAH: 0.0604,
            CSSD: 0.0441,
            KSCM: 0.0365,
            JINA_STRANA: 0.0697
        }
    ),
    'Olomoucký kraj': (
        {
            VOTED: 0.6469,
            NOT_VOTED: 0.3531
        },
        {
            ANO: 0.2977,
            SPOLU: 0.2454,
            PIRSTAN: 0.1235,
            SPD: 0.1224,
            PRISAH: 0.0469,
            CSSD: 0.0453,
            KSCM: 0.0397,
            JINA_STRANA: 0.0791
        }
    ),
    'Zlínský kraj': (
        {
            VOTED: 0.6743,
            NOT_VOTED: 0.3257
        },
        {
            ANO: 0.2698,
            SPOLU: 0.2778,
            PIRSTAN: 0.1344,
            SPD: 0.1138,
            PRISAH: 0.0423,
            CSSD: 0.0489,
            KSCM: 0.0332,
            JINA_STRANA: 0.0798
        }
    ),
    'Moravskoslezský kraj': (
        {
            VOTED: 0.6056,
            NOT_VOTED: 0.3944
        },
        {
            ANO: 0.3373,
            SPOLU: 0.2064,
            PIRSTAN: 0.1113,
            SPD: 0.1282,
            PRISAH: 0.0486,
            CSSD: 0.0542,
            KSCM: 0.0399,
            JINA_STRANA: 0.0741
        }
    )
}