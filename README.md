# Collaborative Optimization of the Age of Information under Partial Observability

python == 3.9.7

requirements: requirements_del_aoi.txt

run main.py file with the following:

    parser = ArgumentParser()
    parser.add_argument('number_of_agents', type=int)
    parser.add_argument('config', type=str)  # {na_dec, na_cen, pomfc, rnd, a1, thr}
    parser.add_argument('state', type=str)  # {t, b}
    parser.add_argument('rnn', type=str)  # {rnn, no}
    parser.add_argument('ch', type=str)  # {known, unknown}
    parser.add_argument('particles', type=str)  # {use, dont}
    parser.add_argument('cont_state', type=str)  # {cont, disc}
    parser.add_argument('weighted_action', type=str)  # {w, u}
    parser.add_argument('drops', type=str)  # {t, f }
    parser.add_argument('--dt', default=None)  # {t, f}
    parser.add_argument('--r', default=None)  # {0.1 ---- 1.0}, {1,2,3,...15}
    parser.add_argument('--true_state_thr', default=None)  # {t,f}

Types of policies:

pomfc:  POMFC-AvgBelief and POMFC-TrueState using state =  {t, b}

na_dec: NA-Dec model both with state =  {t, b}

na_cen: NA model both with state =  {t, b}

rnd: constrant rate policy 

a1: always send policy

thr: threshold policy