window.DEMO_DATA = {
  "trial": 150,
  "systems": [
    {
      "id": "audioseal",
      "name": "AudioSeal",
      "copies": [
        {
          "label": "Clean source",
          "detail": "Unwatermarked",
          "audio": "source_reference",
          "assigned": null,
          "decoded": null,
          "metrics": null
        },
        {
          "label": "Personalized copy A",
          "detail": "Valid copy",
          "audio": "audioseal/member_24199",
          "assigned": 24199,
          "decoded": 24199,
          "metrics": null
        },
        {
          "label": "Personalized copy B",
          "detail": "Valid copy",
          "audio": "audioseal/member_31849",
          "assigned": 31849,
          "decoded": 31849,
          "metrics": null
        },
        {
          "label": "50/50 average",
          "detail": "K = 2",
          "audio": "audioseal/uniform_average",
          "assigned": null,
          "decoded": 23648,
          "coalition": [
            24199,
            31849
          ],
          "metrics": {
            "pesq": 4.6129,
            "stoi": 0.9981,
            "siSdr": 34.65
          }
        }
      ],
      "coalitionSize": [
        {
          "label": "K = 2",
          "detail": "2 copies",
          "coalition": [
            24199,
            31849
          ],
          "decoded": 23648,
          "audio": "conditions/coalition_size/audioseal/k2",
          "metrics": {
            "pesq": 4.6129,
            "stoi": 0.9981,
            "siSdr": 34.65
          }
        },
        {
          "label": "K = 3",
          "detail": "3 copies",
          "coalition": [
            52255,
            17777,
            62914
          ],
          "decoded": 62803,
          "audio": "conditions/coalition_size/audioseal/k3",
          "metrics": {
            "pesq": 4.5913,
            "stoi": 0.9969,
            "siSdr": 32.59
          }
        },
        {
          "label": "K = 5",
          "detail": "5 copies",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 61919,
          "audio": "conditions/coalition_size/audioseal/k5",
          "metrics": {
            "pesq": 4.5748,
            "stoi": 0.9947,
            "siSdr": 31.82
          }
        },
        {
          "label": "K = 8",
          "detail": "8 copies",
          "coalition": [
            2322,
            3134,
            14407,
            21174,
            50027,
            50834,
            51485,
            62890
          ],
          "decoded": 22578,
          "audio": "conditions/coalition_size/audioseal/k8",
          "metrics": {
            "pesq": 4.60428,
            "stoi": 0.997262,
            "siSdr": 34.361277
          }
        }
      ],
      "offsets": [
        {
          "label": "−50 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54239,
          "audio": "conditions/offset/audioseal/minus50",
          "metrics": {
            "pesq": 2.006263,
            "stoi": 0.957199,
            "siSdr": 12.021045
          }
        },
        {
          "label": "−20 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 53727,
          "audio": "conditions/offset/audioseal/minus20",
          "metrics": {
            "pesq": 2.997746,
            "stoi": 0.971549,
            "siSdr": 11.952516
          }
        },
        {
          "label": "−10 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54231,
          "audio": "conditions/offset/audioseal/minus10",
          "metrics": {
            "pesq": 3.995631,
            "stoi": 0.974972,
            "siSdr": 11.909766
          }
        },
        {
          "label": "Aligned",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 61919,
          "audio": "conditions/offset/audioseal/aligned",
          "metrics": {
            "pesq": 4.574791,
            "stoi": 0.994706,
            "siSdr": 31.824466
          }
        },
        {
          "label": "+10 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54231,
          "audio": "conditions/offset/audioseal/plus10",
          "metrics": {
            "pesq": 4.347022,
            "stoi": 0.975642,
            "siSdr": 11.908882
          }
        },
        {
          "label": "+20 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 53727,
          "audio": "conditions/offset/audioseal/plus20",
          "metrics": {
            "pesq": 3.788323,
            "stoi": 0.972689,
            "siSdr": 11.964505
          }
        },
        {
          "label": "+50 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55255,
          "audio": "conditions/offset/audioseal/plus50",
          "metrics": {
            "pesq": 2.75961,
            "stoi": 0.962001,
            "siSdr": 12.017707
          }
        }
      ],
      "codecs": [
        {
          "label": "No codec",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 61919,
          "audio": "conditions/codec/audioseal/none",
          "metrics": {
            "pesq": 4.574791,
            "stoi": 0.994706,
            "siSdr": 31.824466
          }
        },
        {
          "label": "MP3",
          "detail": "128 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 62431,
          "audio": "conditions/codec/audioseal/mp3_128k",
          "metrics": {
            "pesq": 4.375913,
            "stoi": 0.994692,
            "siSdr": 23.017514
          }
        },
        {
          "label": "Opus",
          "detail": "64 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 62431,
          "audio": "conditions/codec/audioseal/opus_64k",
          "metrics": {
            "pesq": 4.525989,
            "stoi": 0.992094,
            "siSdr": 25.686304
          }
        }
      ]
    },
    {
      "id": "wavmark",
      "name": "WavMark",
      "copies": [
        {
          "label": "Clean source",
          "detail": "Unwatermarked",
          "audio": "source_reference",
          "assigned": null,
          "decoded": null,
          "metrics": null
        },
        {
          "label": "Personalized copy A",
          "detail": "Valid copy",
          "audio": "wavmark/member_24199",
          "assigned": 24199,
          "decoded": 24199,
          "metrics": null
        },
        {
          "label": "Personalized copy B",
          "detail": "Valid copy",
          "audio": "wavmark/member_31849",
          "assigned": 31849,
          "decoded": 31849,
          "metrics": null
        },
        {
          "label": "50/50 average",
          "detail": "K = 2",
          "audio": "wavmark/uniform_average",
          "assigned": null,
          "decoded": 32495,
          "coalition": [
            24199,
            31849
          ],
          "metrics": {
            "pesq": 4.6223,
            "stoi": 0.9989,
            "siSdr": 41.2
          }
        }
      ],
      "coalitionSize": [
        {
          "label": "K = 2",
          "detail": "2 copies",
          "coalition": [
            24199,
            31849
          ],
          "decoded": 32495,
          "audio": "conditions/coalition_size/wavmark/k2",
          "metrics": {
            "pesq": 4.6223,
            "stoi": 0.9989,
            "siSdr": 41.2
          }
        },
        {
          "label": "K = 3",
          "detail": "3 copies",
          "coalition": [
            52255,
            17777,
            62914
          ],
          "decoded": 50515,
          "audio": "conditions/coalition_size/wavmark/k3",
          "metrics": {
            "pesq": 4.6005,
            "stoi": 0.9975,
            "siSdr": 39.1
          }
        },
        {
          "label": "K = 5",
          "detail": "5 copies",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/coalition_size/wavmark/k5",
          "metrics": {
            "pesq": 4.5814,
            "stoi": 0.9958,
            "siSdr": 37.09
          }
        },
        {
          "label": "K = 8",
          "detail": "8 copies",
          "coalition": [
            2322,
            3134,
            14407,
            21174,
            50027,
            50834,
            51485,
            62890
          ],
          "decoded": 51518,
          "audio": "conditions/coalition_size/wavmark/k8",
          "metrics": {
            "pesq": 4.464635,
            "stoi": 0.984629,
            "siSdr": 37.989058
          }
        }
      ],
      "offsets": [
        {
          "label": "−50 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55263,
          "audio": "conditions/offset/wavmark/minus50",
          "metrics": {
            "pesq": 1.93093,
            "stoi": 0.960522,
            "siSdr": 12.06977
          }
        },
        {
          "label": "−20 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55263,
          "audio": "conditions/offset/wavmark/minus20",
          "metrics": {
            "pesq": 2.94875,
            "stoi": 0.971522,
            "siSdr": 11.999472
          }
        },
        {
          "label": "−10 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55263,
          "audio": "conditions/offset/wavmark/minus10",
          "metrics": {
            "pesq": 4.00522,
            "stoi": 0.976474,
            "siSdr": 11.964515
          }
        },
        {
          "label": "Aligned",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/offset/wavmark/aligned",
          "metrics": {
            "pesq": 4.581414,
            "stoi": 0.995773,
            "siSdr": 37.091039
          }
        },
        {
          "label": "+10 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55263,
          "audio": "conditions/offset/wavmark/plus10",
          "metrics": {
            "pesq": 4.357479,
            "stoi": 0.977457,
            "siSdr": 11.963194
          }
        },
        {
          "label": "+20 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55263,
          "audio": "conditions/offset/wavmark/plus20",
          "metrics": {
            "pesq": 3.793016,
            "stoi": 0.972636,
            "siSdr": 11.999907
          }
        },
        {
          "label": "+50 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/offset/wavmark/plus50",
          "metrics": {
            "pesq": 2.668235,
            "stoi": 0.963971,
            "siSdr": 12.071367
          }
        }
      ],
      "codecs": [
        {
          "label": "No codec",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/codec/wavmark/none",
          "metrics": {
            "pesq": 4.581414,
            "stoi": 0.995773,
            "siSdr": 37.091039
          }
        },
        {
          "label": "MP3",
          "detail": "128 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/codec/wavmark/mp3_128k",
          "metrics": {
            "pesq": 4.372937,
            "stoi": 0.995762,
            "siSdr": 23.513667
          }
        },
        {
          "label": "Opus",
          "detail": "64 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54751,
          "audio": "conditions/codec/wavmark/opus_64k",
          "metrics": {
            "pesq": 4.509508,
            "stoi": 0.992675,
            "siSdr": 26.636019
          }
        }
      ]
    },
    {
      "id": "timbrewm",
      "name": "TimbreWM",
      "copies": [
        {
          "label": "Clean source",
          "detail": "Unwatermarked",
          "audio": "source_reference",
          "assigned": null,
          "decoded": null,
          "metrics": null
        },
        {
          "label": "Personalized copy A",
          "detail": "Valid copy",
          "audio": "timbrewm/member_378",
          "assigned": 378,
          "decoded": 378,
          "metrics": null
        },
        {
          "label": "Personalized copy B",
          "detail": "Valid copy",
          "audio": "timbrewm/member_497",
          "assigned": 497,
          "decoded": 497,
          "metrics": null
        },
        {
          "label": "50/50 average",
          "detail": "K = 2",
          "audio": "timbrewm/uniform_average",
          "assigned": null,
          "decoded": 504,
          "coalition": [
            378,
            497
          ],
          "metrics": {
            "pesq": 4.5638,
            "stoi": 0.9954,
            "siSdr": 29.58
          }
        }
      ],
      "coalitionSize": [
        {
          "label": "K = 2",
          "detail": "2 copies",
          "coalition": [
            378,
            497
          ],
          "decoded": 504,
          "audio": "conditions/coalition_size/timbrewm/k2",
          "metrics": {
            "pesq": 4.5638,
            "stoi": 0.9954,
            "siSdr": 29.58
          }
        },
        {
          "label": "K = 3",
          "detail": "3 copies",
          "coalition": [
            816,
            277,
            983
          ],
          "decoded": 789,
          "audio": "conditions/coalition_size/timbrewm/k3",
          "metrics": {
            "pesq": 4.3979,
            "stoi": 0.9925,
            "siSdr": 27.48
          }
        },
        {
          "label": "K = 5",
          "detail": "5 copies",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/coalition_size/timbrewm/k5",
          "metrics": {
            "pesq": 4.4677,
            "stoi": 0.9932,
            "siSdr": 26.22
          }
        },
        {
          "label": "K = 8",
          "detail": "8 copies",
          "coalition": [
            36,
            48,
            224,
            329,
            776,
            789,
            802,
            978
          ],
          "decoded": 256,
          "audio": "conditions/coalition_size/timbrewm/k8",
          "metrics": {
            "pesq": 4.560546,
            "stoi": 0.9947,
            "siSdr": 28.650566
          }
        }
      ],
      "offsets": [
        {
          "label": "−50 ms",
          "detail": "Earlier",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/offset/timbrewm/minus50",
          "metrics": {
            "pesq": 2.195064,
            "stoi": 0.956857,
            "siSdr": 11.796238
          }
        },
        {
          "label": "−20 ms",
          "detail": "Earlier",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/offset/timbrewm/minus20",
          "metrics": {
            "pesq": 3.097261,
            "stoi": 0.966797,
            "siSdr": 11.751111
          }
        },
        {
          "label": "−10 ms",
          "detail": "Earlier",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 847,
          "audio": "conditions/offset/timbrewm/minus10",
          "metrics": {
            "pesq": 4.007662,
            "stoi": 0.973016,
            "siSdr": 11.743667
          }
        },
        {
          "label": "Aligned",
          "detail": "Reference",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/offset/timbrewm/aligned",
          "metrics": {
            "pesq": 4.467663,
            "stoi": 0.993202,
            "siSdr": 26.215475
          }
        },
        {
          "label": "+10 ms",
          "detail": "Later",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 847,
          "audio": "conditions/offset/timbrewm/plus10",
          "metrics": {
            "pesq": 4.253011,
            "stoi": 0.973028,
            "siSdr": 11.739361
          }
        },
        {
          "label": "+20 ms",
          "detail": "Later",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 847,
          "audio": "conditions/offset/timbrewm/plus20",
          "metrics": {
            "pesq": 3.693959,
            "stoi": 0.968436,
            "siSdr": 11.748178
          }
        },
        {
          "label": "+50 ms",
          "detail": "Later",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/offset/timbrewm/plus50",
          "metrics": {
            "pesq": 2.743588,
            "stoi": 0.959949,
            "siSdr": 11.809488
          }
        }
      ],
      "codecs": [
        {
          "label": "No codec",
          "detail": "Reference",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/codec/timbrewm/none",
          "metrics": {
            "pesq": 4.467663,
            "stoi": 0.993202,
            "siSdr": 26.215475
          }
        },
        {
          "label": "MP3",
          "detail": "128 kbps",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/codec/timbrewm/mp3_128k",
          "metrics": {
            "pesq": 4.467293,
            "stoi": 0.993197,
            "siSdr": 26.20139
          }
        },
        {
          "label": "Opus",
          "detail": "64 kbps",
          "coalition": [
            640,
            789,
            847,
            846,
            631
          ],
          "decoded": 839,
          "audio": "conditions/codec/timbrewm/opus_64k",
          "metrics": {
            "pesq": 4.418082,
            "stoi": 0.990078,
            "siSdr": 22.591918
          }
        }
      ]
    },
    {
      "id": "voicemark",
      "name": "VoiceMark",
      "copies": [
        {
          "label": "Clean source",
          "detail": "Unwatermarked",
          "audio": "source_reference",
          "assigned": null,
          "decoded": null,
          "metrics": null
        },
        {
          "label": "Personalized copy A",
          "detail": "Valid copy",
          "audio": "voicemark/member_24199",
          "assigned": 24199,
          "decoded": 24199,
          "metrics": null
        },
        {
          "label": "Personalized copy B",
          "detail": "Valid copy",
          "audio": "voicemark/member_31849",
          "assigned": 31849,
          "decoded": 31849,
          "metrics": null
        },
        {
          "label": "50/50 average",
          "detail": "K = 2",
          "audio": "voicemark/uniform_average",
          "assigned": null,
          "decoded": 60600,
          "coalition": [
            24199,
            31849
          ],
          "metrics": {
            "pesq": 3.5749,
            "stoi": 0.9778,
            "siSdr": 14.33
          }
        }
      ],
      "coalitionSize": [
        {
          "label": "K = 2",
          "detail": "2 copies",
          "coalition": [
            24199,
            31849
          ],
          "decoded": 60600,
          "audio": "conditions/coalition_size/voicemark/k2",
          "metrics": {
            "pesq": 3.5749,
            "stoi": 0.9778,
            "siSdr": 14.33
          }
        },
        {
          "label": "K = 3",
          "detail": "3 copies",
          "coalition": [
            52255,
            17777,
            62914
          ],
          "decoded": 64751,
          "audio": "conditions/coalition_size/voicemark/k3",
          "metrics": {
            "pesq": 4.2705,
            "stoi": 0.9891,
            "siSdr": 13.36
          }
        },
        {
          "label": "K = 5",
          "detail": "5 copies",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/coalition_size/voicemark/k5",
          "metrics": {
            "pesq": 3.717,
            "stoi": 0.9725,
            "siSdr": 10.34
          }
        },
        {
          "label": "K = 8",
          "detail": "8 copies",
          "coalition": [
            2322,
            3134,
            21174,
            45243,
            50027,
            50086,
            50834,
            51485
          ],
          "decoded": 51610,
          "audio": "conditions/coalition_size/voicemark/k8",
          "metrics": {
            "pesq": 4.201891,
            "stoi": 0.980679,
            "siSdr": 12.675495
          }
        }
      ],
      "offsets": [
        {
          "label": "−50 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54074,
          "audio": "conditions/offset/voicemark/minus50",
          "metrics": {
            "pesq": 2.036973,
            "stoi": 0.926082,
            "siSdr": 6.714892
          }
        },
        {
          "label": "−20 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54107,
          "audio": "conditions/offset/voicemark/minus20",
          "metrics": {
            "pesq": 2.78873,
            "stoi": 0.937443,
            "siSdr": 6.788801
          }
        },
        {
          "label": "−10 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/offset/voicemark/minus10",
          "metrics": {
            "pesq": 3.246639,
            "stoi": 0.944949,
            "siSdr": 6.619261
          }
        },
        {
          "label": "Aligned",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/offset/voicemark/aligned",
          "metrics": {
            "pesq": 3.716968,
            "stoi": 0.972502,
            "siSdr": 10.342964
          }
        },
        {
          "label": "+10 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/offset/voicemark/plus10",
          "metrics": {
            "pesq": 3.272906,
            "stoi": 0.944067,
            "siSdr": 6.747082
          }
        },
        {
          "label": "+20 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/offset/voicemark/plus20",
          "metrics": {
            "pesq": 3.075757,
            "stoi": 0.939987,
            "siSdr": 6.625187
          }
        },
        {
          "label": "+50 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/offset/voicemark/plus50",
          "metrics": {
            "pesq": 2.412462,
            "stoi": 0.930505,
            "siSdr": 6.718091
          }
        }
      ],
      "codecs": [
        {
          "label": "No codec",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54106,
          "audio": "conditions/codec/voicemark/none",
          "metrics": {
            "pesq": 3.716968,
            "stoi": 0.972502,
            "siSdr": 10.342964
          }
        },
        {
          "label": "MP3",
          "detail": "128 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 37722,
          "audio": "conditions/codec/voicemark/mp3_128k",
          "metrics": {
            "pesq": 3.5764,
            "stoi": 0.972524,
            "siSdr": 10.079962
          }
        },
        {
          "label": "Opus",
          "detail": "64 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 37722,
          "audio": "conditions/codec/voicemark/opus_64k",
          "metrics": {
            "pesq": 3.714983,
            "stoi": 0.971448,
            "siSdr": 10.280356
          }
        }
      ]
    },
    {
      "id": "wmcodec",
      "name": "WMCodec",
      "copies": [
        {
          "label": "Clean source",
          "detail": "Unwatermarked",
          "audio": "source_reference",
          "assigned": null,
          "decoded": null,
          "metrics": null
        },
        {
          "label": "Personalized copy A",
          "detail": "Valid copy",
          "audio": "wmcodec/member_24199",
          "assigned": 24199,
          "decoded": 24199,
          "metrics": null
        },
        {
          "label": "Personalized copy B",
          "detail": "Valid copy",
          "audio": "wmcodec/member_31849",
          "assigned": 31849,
          "decoded": 31849,
          "metrics": null
        },
        {
          "label": "50/50 average",
          "detail": "K = 2",
          "audio": "wmcodec/uniform_average",
          "assigned": null,
          "decoded": 3209,
          "coalition": [
            24199,
            31849
          ],
          "metrics": {
            "pesq": 4.4311,
            "stoi": 0.9861,
            "siSdr": 12.49
          }
        }
      ],
      "coalitionSize": [
        {
          "label": "K = 2",
          "detail": "2 copies",
          "coalition": [
            24199,
            31849
          ],
          "decoded": 3209,
          "audio": "conditions/coalition_size/wmcodec/k2",
          "metrics": {
            "pesq": 4.4311,
            "stoi": 0.9861,
            "siSdr": 12.49
          }
        },
        {
          "label": "K = 3",
          "detail": "3 copies",
          "coalition": [
            52255,
            17777,
            62914
          ],
          "decoded": 34148,
          "audio": "conditions/coalition_size/wmcodec/k3",
          "metrics": {
            "pesq": 4.3938,
            "stoi": 0.9822,
            "siSdr": 11.02
          }
        },
        {
          "label": "K = 5",
          "detail": "5 copies",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/coalition_size/wmcodec/k5",
          "metrics": {
            "pesq": 4.3518,
            "stoi": 0.9803,
            "siSdr": 9.94
          }
        },
        {
          "label": "K = 8",
          "detail": "8 copies",
          "coalition": [
            2322,
            3134,
            14407,
            21174,
            50027,
            50834,
            51485,
            62890
          ],
          "decoded": 52838,
          "audio": "conditions/coalition_size/wmcodec/k8",
          "metrics": {
            "pesq": 4.381025,
            "stoi": 0.978118,
            "siSdr": 10.063621
          }
        }
      ],
      "offsets": [
        {
          "label": "−50 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 42253,
          "audio": "conditions/offset/wmcodec/minus50",
          "metrics": {
            "pesq": 1.878378,
            "stoi": 0.934668,
            "siSdr": 5.859868
          }
        },
        {
          "label": "−20 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55293,
          "audio": "conditions/offset/wmcodec/minus20",
          "metrics": {
            "pesq": 2.733536,
            "stoi": 0.947262,
            "siSdr": 5.827905
          }
        },
        {
          "label": "−10 ms",
          "detail": "Earlier",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 55293,
          "audio": "conditions/offset/wmcodec/minus10",
          "metrics": {
            "pesq": 3.729819,
            "stoi": 0.956081,
            "siSdr": 5.882637
          }
        },
        {
          "label": "Aligned",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/offset/wmcodec/aligned",
          "metrics": {
            "pesq": 4.207767,
            "stoi": 0.979648,
            "siSdr": 9.483491
          }
        },
        {
          "label": "+10 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/offset/wmcodec/plus10",
          "metrics": {
            "pesq": 3.911771,
            "stoi": 0.955588,
            "siSdr": 5.777582
          }
        },
        {
          "label": "+20 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54781,
          "audio": "conditions/offset/wmcodec/plus20",
          "metrics": {
            "pesq": 3.440154,
            "stoi": 0.948097,
            "siSdr": 5.728639
          }
        },
        {
          "label": "+50 ms",
          "detail": "Later",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 54781,
          "audio": "conditions/offset/wmcodec/plus50",
          "metrics": {
            "pesq": 2.372247,
            "stoi": 0.940742,
            "siSdr": 5.891812
          }
        }
      ],
      "codecs": [
        {
          "label": "No codec",
          "detail": "Reference",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/codec/wmcodec/none",
          "metrics": {
            "pesq": 4.207767,
            "stoi": 0.979648,
            "siSdr": 9.483491
          }
        },
        {
          "label": "MP3",
          "detail": "128 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/codec/wmcodec/mp3_128k",
          "metrics": {
            "pesq": 4.20402,
            "stoi": 0.979649,
            "siSdr": 9.479539
          }
        },
        {
          "label": "Opus",
          "detail": "64 kbps",
          "coalition": [
            40973,
            50527,
            54258,
            54171,
            40406
          ],
          "decoded": 1533,
          "audio": "conditions/codec/wmcodec/opus_64k",
          "metrics": {
            "pesq": 4.192534,
            "stoi": 0.977572,
            "siSdr": 9.387513
          }
        }
      ]
    }
  ]
};
